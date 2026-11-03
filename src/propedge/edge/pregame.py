"""Pre-game edge finder (paper only): project every player in a day's games, price every Kalshi
points rung on both sides, and log every would-be flag — flagged or suppressed, with reasons.

Pre-game isn't hurt by our live-feed delay, so this may be the first place a real edge shows up.
Nothing here places bets; flags are stored with is_paper = True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.backtest.points import load_frames
from propedge.data.live_base import InjuryEntry
from propedge.db import models as m
from propedge.db.store import PlayerResolver, utc
from propedge.ids import game_key, name_key, season_for
from propedge.models.bias import bias_factor, role_tier
from propedge.models.kalshi_math import price_both_sides
from propedge.models.points_dist import PointsDistribution, expected_minutes
from propedge.models.points_features import build_features
from propedge.models.points_model import project
from propedge.models.suppression import (QuotePoint, Reason, game_move_reason, market_reasons,
                                         player_move_reason)
from propedge.odds.base import PropQuote

log = logging.getLogger(__name__)


@dataclass
class PregameResult:
    projections: pd.DataFrame      # one row per projected player
    rungs: pd.DataFrame            # one row per (Kalshi contract, side) we could price
    flags: pd.DataFrame            # would-be flags (flagged + suppressed)


# ----------------------------------------------------------------------------- inputs

def _injury_map(session: Session, injuries: list[InjuryEntry]) -> dict[int, InjuryEntry]:
    by_espn = {p.espn_athlete_id: p.id for p in session.scalars(select(m.Player)) if p.espn_athlete_id}
    by_key: dict[str, list[int]] = {}
    for p in session.scalars(select(m.Player)):
        by_key.setdefault(p.name_key, []).append(p.id)
    out = {}
    for inj in injuries:
        pid = by_espn.get(inj.provider_player_id)
        if pid is None and len(by_key.get(name_key(inj.name), [])) == 1:
            pid = by_key[name_key(inj.name)][0]
        if pid is not None:
            out[pid] = inj
    return out


def _targets(session: Session, players: pd.DataFrame, day: date, quotes: list[PropQuote],
             resolver: PlayerResolver) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rows to project for `day`: every Kalshi-listed player plus everyone who played in his
    team's most recent game. Returns (player rows, team rows, quote -> player_id)."""
    games = session.scalars(select(m.Game).where(m.Game.game_date == day)).all()
    season = season_for(day)
    quote_player = {q.ticker: resolver.from_name(q.player_name, q.team) for q in quotes}
    kalshi_team = {quote_player[q.ticker]: q.team for q in quotes if quote_player[q.ticker] and q.team}

    latest = players.sort_values("game_date").groupby("player_id").tail(1).set_index("player_id")
    last_team_game = players.sort_values("game_date").groupby("team")["game_id"].last()
    prow, trow = [], []
    for g in games:
        gid = g.id
        for team, opp, home in ((g.away, g.home, False), (g.home, g.away, True)):
            trow.append(dict(game_id=gid, season=season, game_date=day, team=team, opponent=opp, is_home=home,
                             minutes=np.nan, pts=np.nan, fga=np.nan, fta=np.nan, oreb=np.nan, tov=np.nan))
            ids = {pid for pid, t in kalshi_team.items() if t == team}
            if team in last_team_game.index:
                last = players[players["game_id"] == last_team_game[team]]
                ids |= set(last.loc[last["team"] == team, "player_id"])
            ids = {pid for pid in ids if kalshi_team.get(pid, team) == team}
            for pid in ids:
                if pid not in latest.index:
                    continue            # no NBA history at all: can't project (rookie before his debut)
                prow.append(dict(player_id=pid, game_id=gid, season=season, game_date=day, team=team,
                                 opponent=opp, is_home=home, minutes=np.nan, pts=np.nan))
    return pd.DataFrame(prow), pd.DataFrame(trow), quote_player


# ----------------------------------------------------------------------------- odds history

def _quote_histories(session: Session, tickers: list[str], now: datetime, seconds: float) -> dict[str, list[QuotePoint]]:
    if not tickers:
        return {}
    since = now - timedelta(seconds=seconds)
    rows = session.execute(select(m.OddsSnapshot.ticker, m.OddsSnapshot.first_seen_at, m.OddsSnapshot.yes_bid,
                                  m.OddsSnapshot.yes_ask)
                           .where(m.OddsSnapshot.ticker.in_(tickers), m.OddsSnapshot.last_seen_at >= since,
                                  m.OddsSnapshot.first_seen_at <= now)).all()
    out: dict[str, list[QuotePoint]] = {}
    for t, seen, bid, ask in rows:
        out.setdefault(t, []).append(QuotePoint(utc(seen), bid, ask))
    return out


# ----------------------------------------------------------------------------- main

def season_bias_factors(f_all: pd.DataFrame, params: dict, day: date) -> tuple[dict[int, float], int]:
    """Rolling bias factors (per role tier, or one for everyone) from this season's settled games in the
    window before `day`: uncorrected pre-game projections vs actual points. Settled RESULTS only.
    Returns ({tier: factor}, player-games used)."""
    season = season_for(day)
    since = day - timedelta(days=params["bias_window_days"])
    played = f_all[(f_all["season"] == season) & f_all["pts"].notna() & f_all["ewm_min"].notna()]
    if played.empty:
        return {}, 0
    base = project(played, params)                  # whole team-games (team constraints)
    keep = ((played["game_date"] < day) & (played["game_date"] >= since)).to_numpy()
    if not keep.any():
        return {}, 0
    projected = (base.rate * base.expected_minutes)[keep]
    actual = played["pts"].to_numpy()[keep]
    tiers = role_tier(base.expected_minutes[keep], params["bias_tier_edges"]) if params["bias_by_tier"] \
        else np.zeros(keep.sum(), int)
    return ({int(t): bias_factor(projected[tiers == t], actual[tiers == t], params["bias_prior_games"],
                                 params["bias_cap"]) for t in np.unique(tiers)}, int(keep.sum()))


def run_pregame(session: Session, cfg: dict, day: date, quotes: list[PropQuote], injuries: list[InjuryEntry],
                now: datetime, known_absent_override: dict | None = None, log_flags: bool = True,
                is_mock: bool = False, only_not_started: bool = False, record: bool = True,
                source: str = "run") -> PregameResult:
    """`record` stores every rung's prediction and every player's projection (model_predictions /
    player_projections); `log_flags` stores candidates in `flags`. `source` = "run" for live use,
    "retro" when rebuilding a past day as-of that morning."""
    params = cfg["model"]["points"]
    version = model_version(params)
    seasons = sorted({season_for(day), season_for(day - timedelta(days=365))})
    players, teams, games = load_frames(session, [s for s in cfg["history"]["seasons"] if s <= seasons[-1]])
    players = players[players["game_date"] < day]          # as of the morning of `day`: no look-ahead
    teams = teams[teams["game_date"] < day]
    resolver = PlayerResolver(session)
    quotes = [q for q in quotes if q.game_date == day and q.stat == "pts"]
    if only_not_started:   # pre-game means pre-game: drop games that have tipped off
        started = {g.id for g in session.scalars(select(m.Game).where(m.Game.game_date == day,
                                                                      m.Game.status != "pre"))}
        quotes = [q for q in quotes if game_key(q.game_date, q.away, q.home) not in started]
        if not quotes:
            return PregameResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    tp, tt, quote_player = _targets(session, players, day, quotes, resolver)
    if tp.empty:
        log.warning("pregame: nothing to project for %s", day)
        return PregameResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    injured = _injury_map(session, injuries)
    out_ids = {pid for pid, inj in injured.items() if inj.status.lower() == "out"}
    known_absent = known_absent_override or {}
    if not known_absent_override:
        for (gid, team), grp in tp.groupby(["game_id", "team"]):
            known_absent[(team, gid)] = set(grp["player_id"]) & out_ids
    tp = tp[~tp["player_id"].isin(out_ids)]

    f_all = build_features(pd.concat([players, tp], ignore_index=True), pd.concat([teams, tt], ignore_index=True),
                           games, params, known_absent=known_absent)
    factors, n_bias = season_bias_factors(f_all, params, day)
    f = f_all[(f_all["game_date"] == day) & f_all["pts"].isna() & f_all["ewm_min"].notna()].reset_index(drop=True)
    proj = project(f, params)
    tiers = role_tier(proj.expected_minutes, params["bias_tier_edges"]) if params["bias_by_tier"] \
        else np.zeros(len(f), int)
    factor = np.array([factors.get(int(t), 1.0) for t in tiers])
    proj = replace(proj, rate=proj.rate * factor)
    pmf = proj.pmf()
    names = {p.id: p.name for p in session.scalars(select(m.Player).where(m.Player.id.in_(f["player_id"].tolist())))}
    projections = pd.DataFrame({
        "player_id": f["player_id"], "player": f["player_id"].map(names), "team": f["team"],
        "opponent": f["opponent"], "game_id": f["game_id"],
        "proj_minutes": proj.expected_minutes, "min_mu": proj.min_mu, "min_sd": proj.min_sd, "rate": proj.rate,
        "proj_points": [PointsDistribution(row).mean for row in pmf],
        "points_sd": [PointsDistribution(row).sd for row in pmf],
        "confidence": np.where(proj.low_confidence, "low", "normal"), "low_reasons": proj.low_reasons,
        "injury": f["player_id"].map(lambda pid: injured[pid].status if pid in injured else ""),
        "exp_margin": f["exp_margin"], "margin_source": f["margin_source"], "missing_share": f["missing_share"],
        "pace_factor": f["pace_factor"], "opp_def_factor": f["opp_def_factor"], "bias_factor": factor,
    })
    dist = {pid: PointsDistribution(row) for pid, row in zip(f["player_id"], pmf)}
    row_of = {pid: i for i, pid in enumerate(f["player_id"])}
    log.info("pregame %s: %d players, bias factors by tier %s (from %d settled player-games)", day, len(f),
             {k: round(v, 3) for k, v in factors.items()}, n_bias)

    # Suppression context: price histories for every contract in these games.
    nf = cfg["edge"]["no_flag"]
    window = max(nf["after_price_move_s"], nf["after_game_move_s"]) + nf["price_move_lookback_s"] + 60
    hist = _quote_histories(session, [q.ticker for q in quotes], now, window)
    by_game_player: dict[str, dict[int, dict[str, list[QuotePoint]]]] = {}
    for q in quotes:
        pid = quote_player.get(q.ticker)
        if pid is not None and q.ticker in hist:
            gid = game_key(q.game_date, q.away, q.home)
            by_game_player.setdefault(gid, {}).setdefault(pid, {})[q.ticker] = hist[q.ticker]
    game_reason = {gid: game_move_reason(players_h, now, cfg) for gid, players_h in by_game_player.items()}

    fees, ed = cfg["odds"]["kalshi"]["fees"], cfg["edge"]
    rungs, cands = [], []
    for q in quotes:
        pid = quote_player.get(q.ticker)
        if pid not in dist:
            continue
        gid = game_key(q.game_date, q.away, q.home)
        p_yes = dist[pid].p_over(q.line)
        i = row_of[pid]
        conf = "low" if proj.low_confidence[i] else "normal"
        rungs.append(dict(ticker=q.ticker, player_id=pid, player=names.get(pid), game_id=gid, line=q.line,
                          side="yes", model_prob=p_yes, yes_bid=q.yes_bid, yes_ask=q.yes_ask,
                          proj_mean=dist[pid].mean, confidence=conf))
        for side in price_both_sides(p_yes, q.yes_bid, q.yes_ask, cfg["odds"]["kalshi"]["order_contracts"],
                                     fees["taker_coef"], 1.0, fees["round_to"]):
            if side.edge < ed["log_min_edge"]:
                continue
            reasons: list[Reason] = []
            if proj.low_confidence[i]:
                reasons.append(Reason("low_confidence", proj.low_reasons[i]))
            if pid in injured:
                reasons.append(Reason("injury_status", f"{injured[pid].status}: {injured[pid].comment[:80]}"))
            reasons += market_reasons(q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size, side.side,
                                      q.fetched_at, now, cfg)
            if (r := player_move_reason(by_game_player.get(gid, {}).get(pid, {}), now, cfg)):
                reasons.append(r)
            if game_reason.get(gid):
                reasons.append(game_reason[gid])
            inputs = {
                "proj": {k: _num(projections.iloc[i][k]) for k in
                         ("proj_minutes", "min_sd", "rate", "proj_points", "points_sd", "exp_margin",
                          "pace_factor", "opp_def_factor", "missing_share", "bias_factor")},
                "margin_source": projections.iloc[i]["margin_source"], "low_reasons": proj.low_reasons[i],
                "rate_cv": _num(proj.rate_cv[i]), "phi": proj.phi,
                "quote": {"yes_bid": q.yes_bid, "yes_ask": q.yes_ask, "yes_bid_size": q.yes_bid_size,
                          "yes_ask_size": q.yes_ask_size, "fetched_at": q.fetched_at.isoformat()},
                "fees": {"coef": fees["taker_coef"], "contracts": cfg["odds"]["kalshi"]["order_contracts"]},
                "params": {k: _num(v) for k, v in params.items()},
                "model_version": version,
            }
            passes = side.edge >= ed["min_edge"] and side.ev_per_dollar >= ed["min_ev_per_dollar"]
            cands.append(dict(ticker=q.ticker, player_id=pid, player=names.get(pid), game_id=gid, line=q.line,
                              side=side.side, model_prob=side.model_prob, yes_bid=q.yes_bid, yes_ask=q.yes_ask,
                              ask=side.ask, fee=side.fee, cost=side.cost, edge=side.edge,
                              ev_per_dollar=side.ev_per_dollar, confidence=conf, passes=passes,
                              reasons=[r.as_dict() for r in reasons], inputs=inputs))

    flags_df = pd.DataFrame(cands)
    if not flags_df.empty:
        flags_df["status"] = assign_statuses(flags_df, ed["one_flag_per_player"])
        flags_df = flags_df.drop(columns=["passes"])
    if record:
        _record(session, projections, pd.DataFrame(rungs), now, source, version, is_mock)
    if log_flags and not flags_df.empty:
        _log_flags(session, flags_df, now, is_mock, ed["pregame_flag_cooldown_min"])
    return PregameResult(projections, pd.DataFrame(rungs), flags_df)


def assign_statuses(c: pd.DataFrame, one_per_player: bool) -> list[str]:
    """flag / suppressed / not_best / below_threshold for each candidate row (see models.Flag)."""
    status = []
    best = {}
    if one_per_player:
        for pid, grp in c[c["passes"]].groupby("player_id"):
            best[pid] = grp["edge"].idxmax()
    for idx, row in c.iterrows():
        if not row["passes"]:
            status.append("below_threshold")
        elif row["reasons"]:
            status.append("suppressed")
        elif one_per_player and best.get(row["player_id"]) != idx:
            status.append("not_best")
        else:
            status.append("flag")
    return status


def _record(session: Session, projections: pd.DataFrame, rungs: pd.DataFrame, now: datetime, source: str,
            version: str, is_mock: bool) -> None:
    for r in projections.to_dict("records"):
        session.add(m.PlayerProjection(created_at=now, source=source, game_id=r["game_id"], player_id=int(r["player_id"]),
                                       proj_minutes=float(r["proj_minutes"]), proj_points=float(r["proj_points"]),
                                       points_sd=float(r["points_sd"]), bias_factor=float(r["bias_factor"]),
                                       confidence=r["confidence"], low_reasons=r["low_reasons"] or "",
                                       model_version=version))
    for r in rungs.to_dict("records"):
        session.add(m.ModelPrediction(created_at=now, kind="pregame", source=source, game_id=r["game_id"],
                                      player_id=int(r["player_id"]), ticker=r["ticker"], line=float(r["line"]),
                                      p_yes=float(r["model_prob"]), proj_mean=float(r["proj_mean"]),
                                      confidence=r["confidence"], model_version=version, is_mock=is_mock))
    session.commit()


def actual_absences(session: Session, cfg: dict, day: date) -> dict[tuple[str, str], set]:
    """For a day already played: rotation players (≥ missing_min_mpg in one of the team's last
    `missing_recent_games` games) who didn't play — a stand-in for that day's injury report."""
    params = cfg["model"]["points"]
    players, _, _ = load_frames(session, [season_for(day - timedelta(days=200)), season_for(day)])
    out = {}
    for (team, gid), grp in players[players["game_date"] == day].groupby(["team", "game_id"]):
        before = players[(players["team"] == team) & (players["game_date"] < day)]
        recent_games = before.sort_values("game_date")["game_id"].drop_duplicates().tail(params["missing_recent_games"])
        rot = before[before["game_id"].isin(recent_games) & (before["minutes"] >= params["missing_min_mpg"])]
        out[(team, gid)] = set(rot["player_id"]) - set(grp["player_id"])
    return out


def settle(session: Session, flags: pd.DataFrame, day: date) -> pd.DataFrame:
    """Paper results for a finished day: win/loss per would-be flag, P&L per contract.
    Players who didn't play are 'void' (Kalshi settles those at a fair price)."""
    if flags.empty:
        return flags
    pts = dict(session.execute(select(m.PlayerGameLog.player_id, m.PlayerGameLog.pts)
                               .where(m.PlayerGameLog.game_date == day)).all())
    out = flags.copy()
    out["actual_pts"] = out["player_id"].map(pts)
    yes_won = out["actual_pts"] > out["line"]
    won = np.where(out["side"] == "yes", yes_won, ~yes_won)
    out["result"] = np.where(out["actual_pts"].isna(), "void", np.where(won, "win", "loss"))
    out["pnl_per_contract"] = np.where(out["result"] == "void", 0.0,
                                       np.where(out["result"] == "win", 1 - out["cost"], -out["cost"]))
    return out


def model_version(params: dict) -> str:
    """Short fingerprint of the model parameters, stored with every flag (results can be split by version)."""
    import hashlib
    import json
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)


def _log_flags(session: Session, flags: pd.DataFrame, now: datetime, is_mock: bool, cooldown_min: float) -> int:
    """Store would-be flags. Skip a repeat of the same contract/side/status within the cooldown
    unless the edge changed by more than 2 points (keeps the periodic job from spamming rows)."""
    since = now - timedelta(minutes=cooldown_min)
    recent = {(r.ticker, r.side, r.status): r.edge for r in session.scalars(
        select(m.Flag).where(m.Flag.kind == "pregame", m.Flag.created_at >= since, m.Flag.created_at <= now))}
    n = 0
    for row in flags.to_dict("records"):
        prev = recent.get((row["ticker"], row["side"], row["status"]))
        if prev is not None and abs(prev - row["edge"]) < 0.02:
            continue
        session.add(m.Flag(created_at=now, kind="pregame", status=row["status"], reasons=row["reasons"],
                           is_paper=True, game_id=row["game_id"], player_id=int(row["player_id"]),
                           ticker=row["ticker"], stat="pts", line=row["line"], side=row["side"],
                           model_prob=row["model_prob"], ask=row["ask"], fee_per_contract=row["fee"],
                           cost=row["cost"], edge=row["edge"], ev_per_dollar=row["ev_per_dollar"],
                           confidence=row["confidence"], inputs=row["inputs"], is_mock=is_mock))
        n += 1
    session.commit()
    counts = flags["status"].value_counts().to_dict()
    yes_no = flags[flags["status"] == "flag"]["side"].value_counts().to_dict()
    log.info("pregame: logged %d candidates %s; flags by side %s", n, counts, yes_no)
    return n
