"""Pre-game features for the points model, built with NO look-ahead. Pure pandas, no I/O.

Rule: every feature for a game on date D uses only games before D. That's done with
`.shift(1)` after sorting by date (the current game is never in its own average), and for
league-wide numbers by only using days strictly before D.

Inputs (DataFrames, one row per player-game / team-game; rows to project — e.g. tonight's
games — are included with pts/minutes = NaN):
    players: player_id, game_id, season, game_date, team, opponent, is_home, minutes, pts
    teams:   game_id, season, game_date, team, opponent, is_home, minutes, pts, fga, fta, oreb, tov
    games:   game_id, home_spread (may be NaN)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- team level

def _league_before(df: pd.DataFrame, value: str, weight: str) -> pd.Series:
    """For each row: league-wide weighted mean of `value` over all rows on EARLIER dates of the
    same season (falls back to the previous season's average early on)."""
    daily = df.groupby(["season", "game_date"]).apply(
        lambda g: pd.Series({"v": (g[value] * g[weight]).sum(), "w": g[weight].sum()}), include_groups=False)
    daily = daily.reset_index().sort_values(["season", "game_date"])
    daily[["cv", "cw"]] = daily.groupby("season")[["v", "w"]].cumsum() - daily[["v", "w"]]
    season_avg = daily.groupby("season").apply(lambda g: g["v"].sum() / g["w"].sum(), include_groups=False)
    prev = season_avg.shift(1)
    daily["avg"] = daily["cv"] / daily["cw"].replace(0, np.nan)
    few = daily["cw"] < daily["w"].median() * 3            # first ~3 nights: borrow last season's average
    daily.loc[few, "avg"] = daily.loc[few, "season"].map(prev).fillna(daily.loc[few, "avg"])
    return df[["season", "game_date"]].merge(daily[["season", "game_date", "avg"]], how="left")["avg"].to_numpy()


def team_features(teams: pd.DataFrame, games: pd.DataFrame, shrink_games: float, home_adv: float) -> pd.DataFrame:
    """Per team-game: season-to-date pace, offensive and defensive ratings (shrunk toward the league),
    and the expected margin (real closing spread when we have it, otherwise a net-rating estimate)."""
    t = teams.copy().sort_values(["team", "game_date"]).reset_index(drop=True)
    t["poss"] = t["fga"] + 0.44 * t["fta"] - t["oreb"] + t["tov"]
    opp = t[["game_id", "team", "pts", "poss"]].rename(columns={"team": "opponent", "pts": "opp_pts", "poss": "opp_poss"})
    t = t.merge(opp, on=["game_id", "opponent"], how="left")
    t["game_poss"] = (t["poss"] + t["opp_poss"]) / 2
    t["pace48"] = t["game_poss"] / (t["minutes"] / 5) * 48
    t["ortg"] = 100 * t["pts"] / t["game_poss"]
    t["drtg"] = 100 * t["opp_pts"] / t["game_poss"]
    t = t.sort_values(["team", "game_date"]).reset_index(drop=True)

    played = t["pts"].notna().astype(float)
    t["_one"] = played
    for col in ("pace48", "ortg", "drtg"):
        t[f"_{col}"] = t[col].fillna(0) * played
    g = t.groupby(["team", "season"])
    n_before = g["_one"].cumsum() - t["_one"]
    league = {}
    for col in ("pace48", "ortg", "drtg"):
        league[col] = _league_before(t[t["pts"].notna()].assign(_w=1.0), col, "_w")
    # map league averages (computed on played rows) back to every row by date
    lg = t[t["pts"].notna()][["season", "game_date"]].assign(**{f"lg_{c}": v for c, v in league.items()})
    lg = lg.drop_duplicates(["season", "game_date"])
    t = t.merge(lg, on=["season", "game_date"], how="left")
    t = t.sort_values(["season", "game_date"])
    for c in league:   # rows to project (tonight) have no same-day league value: carry the latest forward
        t[f"lg_{c}"] = t.groupby("season")[f"lg_{c}"].ffill()
    t = t.sort_values(["team", "game_date"]).reset_index(drop=True)
    g = t.groupby(["team", "season"])
    n_before = g["_one"].cumsum() - t["_one"]
    for col in ("pace48", "ortg", "drtg"):
        sum_before = g[f"_{col}"].cumsum() - t[f"_{col}"]
        # Shrinkage: act as if we'd also seen `shrink_games` league-average games.
        t[f"{col}_pre"] = (sum_before + shrink_games * t[f"lg_{col}"]) / (n_before + shrink_games)
    t["games_before"] = n_before

    # Opponent's pre-game numbers on the same row.
    o = t[["game_id", "team", "pace48_pre", "ortg_pre", "drtg_pre"]].rename(columns={
        "team": "opponent", "pace48_pre": "opp_pace48_pre", "ortg_pre": "opp_ortg_pre", "drtg_pre": "opp_drtg_pre"})
    t = t.merge(o, on=["game_id", "opponent"], how="left")
    cols = ["game_id", "home_spread"] + (["total"] if "total" in games else [])
    t = t.merge(games[cols], on="game_id", how="left")
    t["home_spread"] = pd.to_numeric(t["home_spread"], errors="coerce")
    t["total"] = pd.to_numeric(t["total"], errors="coerce") if "total" in t else np.nan

    exp_pace = (t["pace48_pre"] + t["opp_pace48_pre"]) / 2
    net_diff = (t["ortg_pre"] - t["drtg_pre"]) - (t["opp_ortg_pre"] - t["opp_drtg_pre"])
    proxy_margin = net_diff * exp_pace / 100 + np.where(t["is_home"], home_adv, -home_adv)
    spread_margin = np.where(t["is_home"], -t["home_spread"], t["home_spread"])
    t["exp_margin"] = np.where(t["home_spread"].notna(), spread_margin, proxy_margin)
    t["margin_source"] = np.where(t["home_spread"].notna(), "spread", "net_rating")
    t["pace_factor"] = exp_pace / t["lg_pace48"]
    # Team's expected points: from the game's betting total and spread when we have them
    # (home = total/2 − spread/2), else from ratings × expected possessions.
    line_pts = t["total"] / 2 + np.where(t["is_home"], -t["home_spread"], t["home_spread"]) / 2
    rating_pts = (t["ortg_pre"] * t["opp_drtg_pre"] / t["lg_drtg"]) * exp_pace / 100
    t["team_exp_pts"] = np.where(t["total"].notna() & t["home_spread"].notna(), line_pts, rating_pts)
    # How generous the opponent's defence is relative to the league (points allowed per 100 possessions).
    t["opp_def_factor"] = t["opp_drtg_pre"] / t["lg_drtg"]
    return t[["game_id", "team", "exp_margin", "margin_source", "pace_factor", "opp_def_factor", "games_before",
              "team_exp_pts"]]


# ----------------------------------------------------------------------------- teammates out

def newly_missing_share(players: pd.DataFrame, recent_games: int, min_mpg: float,
                        known_absent: dict[tuple[str, str], set] | None = None) -> pd.Series:
    """Per team-game: share of the team's scoring that is *newly* missing — rotation players
    (≥ `min_mpg` recently) who played in one of the team's last `recent_games` games but not
    in this one. Long-term absences aren't counted: teammates' recent stats already reflect them.

    Uses actual absences, which in real life are known pre-game from the injury report (late
    scratches are the exception — a small optimistic bias, noted in the report).
    For games not played yet, absences come from `known_absent[(team, game_id)]` (injury report)."""
    p = players[players["minutes"].notna()]
    team_games = (players[["team", "game_id", "game_date"]].drop_duplicates()
                  .sort_values(["team", "game_date"]))
    roster_by_game = p.groupby(["team", "game_id"])["player_id"].apply(set).to_dict()
    ppg = p.sort_values("game_date").groupby("player_id")["pts"].apply(lambda s: s.ewm(halflife=10).mean())
    p = p.assign(_ppg=ppg.reset_index(level=0, drop=True).reindex(p.index))
    mpg_by = p.set_index(["team", "game_id", "player_id"])[["minutes", "_ppg"]].to_dict("index")

    out = {}
    for team, tg in team_games.groupby("team"):
        history: list[str] = []
        for gid in tg["game_id"]:
            recent = {}
            for prev in history[-recent_games:]:
                for pid in roster_by_game.get((team, prev), set()):
                    row = mpg_by.get((team, prev, pid))
                    if row and row["minutes"] >= min_mpg:
                        recent[pid] = row["_ppg"]
            team_pts = sum(recent.values()) or np.nan
            if (team, gid) in roster_by_game:
                present = roster_by_game[(team, gid)]
                missing = sum(v for pid, v in recent.items() if pid not in present)
            else:   # not played yet: only the players the injury report says are out
                absent = (known_absent or {}).get((team, gid), set())
                missing = sum(v for pid, v in recent.items() if pid in absent)
            out[(team, gid)] = missing / team_pts if team_pts == team_pts else 0.0
            if (team, gid) in roster_by_game:
                history.append(gid)
    return pd.Series(out, name="missing_share")


# ----------------------------------------------------------------------------- player level

def seeded_season_ewm(values: np.ndarray, season: np.ndarray, seed: np.ndarray, halflife: float,
                      seed_games: float) -> np.ndarray:
    """Pre-game exponentially weighted average that RESTARTS each season, seeded with `seed`
    (e.g. last season's minutes per game) counted as `seed_games` games.

    Why: at the start of a season the last few games of LAST season are a bad guide — stars rest,
    sit in blowouts or play hurt in April. Values must be one player's rows in date order; NaN
    values (games not played yet) don't update the average. Output for each row uses only
    earlier rows."""
    decay = 0.5 ** (1 / halflife)
    out = np.full(len(values), np.nan)
    total = weight = 0.0
    current = None
    for i, (v, s, sd) in enumerate(zip(values, season, seed)):
        if s != current:
            current = s
            total, weight = (sd * seed_games, seed_games) if seed_games > 0 and not np.isnan(sd) else (0.0, 0.0)
        out[i] = total / weight if weight > 0 else np.nan
        if not np.isnan(v):
            total, weight = total * decay + v, weight * decay + 1
    return out


def player_features(players: pd.DataFrame, min_halflife: float, rate_halflife: float,
                    new_team_games: int, rookie_games: int, season_seed_games: float = 0) -> pd.DataFrame:
    """Per player-game history features (all computed from earlier games only)."""
    p = players.copy().sort_values(["player_id", "game_date"]).reset_index(drop=True)
    g = p.groupby("player_id")
    played = p["minutes"].notna()

    def ewm_shift(col: str, halflife: float) -> pd.Series:
        return g[col].transform(lambda s: s.ewm(halflife=halflife, ignore_na=True).mean().shift(1))

    # Minutes: restart each season from last season's minutes per game (see seeded_season_ewm).
    prev_mpg = (p[played].groupby(["player_id", "season"])["minutes"].mean().rename("_prev_mpg").reset_index())
    seasons_sorted = sorted(p["season"].unique())
    prev_mpg["season"] = prev_mpg["season"].map(dict(zip(seasons_sorted, seasons_sorted[1:])))
    p = p.merge(prev_mpg.dropna(subset=["season"]), on=["player_id", "season"], how="left")
    p["ewm_min"] = np.concatenate([
        seeded_season_ewm(grp["minutes"].to_numpy(float), grp["season"].to_numpy(), grp["_prev_mpg"].to_numpy(float),
                          min_halflife, season_seed_games)
        for _, grp in p.groupby("player_id", sort=True)])
    # A player with no history this season and none last season: fall back to his last games.
    p["ewm_min"] = p["ewm_min"].fillna(ewm_shift("minutes", min_halflife))
    g = p.groupby("player_id")
    p["ewm_min_sd"] = g["minutes"].transform(lambda s: s.ewm(halflife=min_halflife, ignore_na=True).std().shift(1))
    p["ewm_pts_r"] = ewm_shift("pts", rate_halflife)
    p["ewm_min_r"] = ewm_shift("minutes", rate_halflife)
    p["rate_obs"] = p["ewm_pts_r"] / p["ewm_min_r"].replace(0, np.nan)
    p["career_min"] = g["minutes"].transform(lambda s: s.fillna(0).cumsum().shift(1).fillna(0))
    p["career_games"] = g["minutes"].transform(lambda s: s.notna().cumsum().shift(1).fillna(0))

    # Baselines (what the model must beat).
    gs = p.groupby(["player_id", "season"])
    p["season_games"] = gs["pts"].transform(lambda s: s.notna().cumsum().shift(1).fillna(0))
    p["season_avg"] = gs["pts"].transform(lambda s: s.expanding().mean().shift(1))
    last_season = p[played].groupby(["player_id", "season"])["pts"].mean().rename("prev_season_avg").reset_index()
    seasons = sorted(p["season"].unique())
    last_season["season"] = last_season["season"].map({s: n for s, n in zip(seasons, seasons[1:])})
    p = p.merge(last_season.dropna(subset=["season"]), on=["player_id", "season"], how="left")
    p["baseline_season_avg"] = p["season_avg"].where(p["season_games"] > 0, p["prev_season_avg"])
    p["baseline_last10"] = p.groupby("player_id")["pts"].transform(
        lambda s: s.rolling(10, min_periods=1).mean().shift(1))

    # Rest and role changes.
    p["prev_date"] = p.groupby("player_id")["game_date"].shift(1)
    p["days_rest"] = (pd.to_datetime(p["game_date"]) - pd.to_datetime(p["prev_date"])).dt.days
    p["prev_team"] = p.groupby("player_id")["team"].shift(1)
    p["team_change"] = p["prev_team"].notna() & (p["team"] != p["prev_team"])
    p["_stint"] = p.groupby("player_id")["team_change"].cumsum()
    p["games_with_team"] = p.groupby(["player_id", "_stint"]).cumcount()
    p["new_team"] = (p["_stint"] > 0) & (p["games_with_team"] < new_team_games)
    first_season = p.groupby("player_id")["season"].transform("min")
    p["rookie"] = (p["season"] == first_season) & (p["season"] != seasons[0]) & \
                  (p["career_games"] < rookie_games)
    p["history_unknown"] = (p["season"] == seasons[0]) & (p["career_games"] < rookie_games)
    return p


def missed_team_games(players: pd.DataFrame, teams: pd.DataFrame, return_missed_games: int,
                      return_window_games: int) -> pd.Series:
    """True for a player's first `return_window_games` games after missing ≥ `return_missed_games`
    of his team's games in a row (back from injury/absence)."""
    tg = teams[["team", "game_date"]].drop_duplicates().sort_values(["team", "game_date"])
    idx = {team: pd.to_datetime(d["game_date"]).to_numpy() for team, d in tg.groupby("team")}
    p = players.sort_values(["player_id", "game_date"])
    prev = p.groupby("player_id")["game_date"].shift(1)
    prev_team = p.groupby("player_id")["team"].shift(1)
    missed = []
    for team, d, pd_, pt in zip(p["team"], p["game_date"], prev, prev_team):
        if pd.isna(pd_) or pt != team or team not in idx:
            missed.append(0)
            continue
        dates = idx[team]
        missed.append(int(((dates > np.datetime64(pd_)) & (dates < np.datetime64(d))).sum()))
    flag = pd.Series(np.array(missed) >= return_missed_games, index=p.index)
    returning = flag.groupby(p["player_id"]).transform(
        lambda s: s.astype(float).rolling(return_window_games, min_periods=1).max().astype(bool))
    return returning.reindex(players.index).fillna(False)


def build_features(players: pd.DataFrame, teams: pd.DataFrame, games: pd.DataFrame, params: dict,
                   known_absent: dict[tuple[str, str], set] | None = None) -> pd.DataFrame:
    """All features for every player-game row (history rows and rows to project)."""
    pf = player_features(players, params["min_halflife_games"], params["rate_halflife_games"],
                         params["new_team_games"], params["rookie_games"], params["season_seed_games"])
    pf["returning"] = missed_team_games(pf, teams, params["return_missed_games"],
                                        params["return_window_games"]).to_numpy()
    tf = team_features(teams, games, params["team_shrink_games"], params["home_adv_points"])
    pf = pf.merge(tf, on=["game_id", "team"], how="left")
    miss = newly_missing_share(players, params["missing_recent_games"], params["missing_min_mpg"], known_absent)
    pf = pf.merge(miss.rename_axis(["team", "game_id"]).reset_index(), on=["team", "game_id"], how="left")
    pf["missing_share"] = pf["missing_share"].fillna(0.0)
    # Share of this team-game's (pre-game expected) scoring that belongs to players new to the team.
    # Big arrivals reshuffle everyone's role (e.g. a star joins): the market prices that, our
    # history-based rates can't yet — so teammates get low confidence early in the season.
    exp_pts = (pf["ewm_pts_r"].fillna(0)).where(pf["ewm_pts_r"] > 0, 0)
    tot = exp_pts.groupby([pf["team"], pf["game_id"]]).transform("sum")
    new = (exp_pts * pf["new_team"]).groupby([pf["team"], pf["game_id"]]).transform("sum")
    pf["team_new_share"] = (new / tot.replace(0, np.nan)).fillna(0.0)
    return pf
