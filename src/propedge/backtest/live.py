"""Live-engine backtest on past games rebuilt from play-by-play.

For each game and each sample moment (config live.sample_points, in game minutes):
  - the moment's real (wall-clock) time τ is the time of the play that reached that game minute;
  - the model sees the game as it was at τ − assumed_feed_delay_s (our ESPN delay), not at τ;
  - every player's still-open ladder lines (10+, 15+, …) are priced and scored against the final.
Compared with: the static pre-game probability, and a naive projection (points so far + pre-game rate
× remaining share of his pre-game minutes, Poisson-like spread). Optionally Kalshi's mid at τ.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from propedge.backtest.points import load_frames
from propedge.data.live_base import LivePlay
from propedge.db import models as m
from propedge.db.store import utc
from propedge.models.bias import role_tier, rolling_bias_factors
from propedge.models.game_state import elapsed_minutes, replay_states
from propedge.models.live_points import LiveInputs, simulate
from propedge.models.points_dist import PointsDistribution
from propedge.models.points_features import build_features
from propedge.models.points_model import project
from propedge.models.scoring import brier, log_loss

log = logging.getLogger(__name__)
LADDER = np.array([9.5, 14.5, 19.5, 24.5, 29.5, 34.5, 39.5])


# ----------------------------------------------------------------------------- loading

@dataclass
class GameData:
    game_id: str
    game_date: object
    away: str
    home: str
    plays: list[LivePlay]
    starters: dict[str, set[str]]
    team_of: dict[str, str]
    final_pts: dict[str, int]
    final_min: dict[str, float]


def load_game(session: Session, game_id: str) -> GameData | None:
    game = session.get(m.Game, game_id)
    rows = session.scalars(select(m.Play).where(m.Play.game_id == game_id).order_by(m.Play.sequence)).all()
    if not rows or game is None:
        return None
    plays = [LivePlay(r.provider_play_id, r.sequence, r.period, r.clock_s, utc(r.wallclock) if r.wallclock else None,
                      r.play_type, r.text, r.team, r.scoring, r.score_value, r.away_score, r.home_score,
                      tuple(str(i) for i in (r.player_ids or []))) for r in rows]
    first_snap = session.scalar(select(func.min(m.LivePlayerSnapshot.snapshot_id)).join(
        m.LiveGameSnapshot, m.LiveGameSnapshot.id == m.LivePlayerSnapshot.snapshot_id).where(
        m.LiveGameSnapshot.game_id == game_id))
    box = session.scalars(select(m.LivePlayerSnapshot).where(m.LivePlayerSnapshot.snapshot_id == first_snap)).all()
    team_of = {str(b.player_id): b.team for b in box}
    starters: dict[str, set[str]] = {}
    for b in box:
        if b.starter:
            starters.setdefault(b.team, set()).add(str(b.player_id))
    logs = session.execute(select(m.PlayerGameLog.player_id, m.PlayerGameLog.pts, m.PlayerGameLog.minutes)
                           .where(m.PlayerGameLog.game_id == game_id)).all()
    final_pts = {str(pid): pts for pid, pts, _ in logs}
    final_min = {str(pid): mins for pid, _, mins in logs}
    if not final_pts or sum(len(v) for v in starters.values()) < 10:
        return None
    return GameData(game_id, game.game_date, game.away, game.home, plays, starters, team_of, final_pts, final_min)


def pregame_table(session: Session, cfg: dict, seasons: list[str]) -> pd.DataFrame:
    """Pre-game projection for every played player-game in `seasons` (no look-ahead), incl. the
    rolling bias factor and each player's foul rate before the game."""
    params = cfg["model"]["points"]
    players, teams, games = load_frames(session, cfg["history"]["seasons"])
    f = build_features(players, teams, games, params)
    f = f[f["season"].isin(seasons) & f["ewm_min"].notna() & f["pts"].notna()].reset_index(drop=True)
    proj = project(f, params)
    base = proj.rate * proj.expected_minutes
    tiers = role_tier(proj.expected_minutes, params["bias_tier_edges"]) if params["bias_by_tier"] else None
    fac = rolling_bias_factors(f["season"], f["game_date"], base, f["pts"].to_numpy(), params["bias_prior_games"],
                               params["bias_cap"], params["bias_window_days"], tiers)
    pmf = proj.pmf()
    pf = pd.DataFrame(session.execute(select(m.PlayerGameLog.player_id, m.PlayerGameLog.game_date,
                                             m.PlayerGameLog.pf, m.PlayerGameLog.minutes)).all(),
                      columns=["player_id", "game_date", "pf", "minutes"]).sort_values(["player_id", "game_date"])
    g = pf.groupby("player_id")
    pf["pf_before"] = g["pf"].cumsum() - pf["pf"]
    pf["min_before"] = g["minutes"].cumsum() - pf["minutes"]
    lv = cfg["live"]
    pf["pf_rate"] = (pf["pf_before"] + lv["pf_rate_league"] * lv["pf_rate_prior_min"]) / \
                    (pf["min_before"] + lv["pf_rate_prior_min"])
    f = f.merge(pf[["player_id", "game_date", "pf_rate"]], on=["player_id", "game_date"], how="left")
    return pd.DataFrame({
        "game_id": f["game_id"], "player_id": f["player_id"].astype(str), "rate_pre": proj.rate * fac,
        "min_pre": proj.min_mu, "min_sd": proj.min_sd, "low_conf": proj.low_confidence,
        "pf_rate": f["pf_rate"].fillna(lv["pf_rate_league"]),
        "pmf": list(pmf * 1.0),     # pre-game distribution (without bias factor; small effect)
    })


# ----------------------------------------------------------------------------- snapshots

def sample_moments(g: GameData, sample_points: list[float], delay_s: float) -> list[tuple[float, object, float]]:
    """[(true game minute, wall time τ, game minute the model sees at τ − delay)]."""
    timed = [(elapsed_minutes(p.period, p.clock_s), p.wallclock) for p in g.plays if p.wallclock is not None]
    timed.sort(key=lambda t: (t[1], t[0]))
    out = []
    for T in sample_points:
        reach = [w for e, w in timed if e >= T]
        if not reach:
            continue
        tau = min(reach)
        seen = [e for e, w in timed if w <= tau - timedelta(seconds=delay_s)]
        if seen:
            out.append((T, tau, max(seen)))
    return out


def snapshot_rows(g: GameData, pre: pd.DataFrame, p: dict, n_sims: int, rng: np.random.Generator,
                  delay_s: float, sample_points: list[float], kalshi_mid=None) -> list[dict]:
    """`kalshi_mid(player_id, line, tau)` -> Kalshi's mid at real time tau (or None), optional."""
    """Model / pre-game / naive probabilities for every open ladder line, at every sample moment."""
    pre_g = pre[pre["game_id"] == g.game_id].set_index("player_id")
    moments = sample_moments(g, sample_points, delay_s)
    if pre_g.empty or not moments:
        return []
    states = replay_states(g.plays, g.starters, g.team_of, [seen for _, _, seen in moments])
    rows = []
    for (T, tau, seen), st in zip(moments, states):
        ids = [pid for pid in pre_g.index if pid in st.players and pid in g.final_pts and g.final_min.get(pid, 0) > 0]
        if not ids:
            continue
        ps = [st.players[i] for i in ids]
        margin_home = st.home_score - st.away_score
        x = LiveInputs(
            pts=np.array([s.pts for s in ps], float), fouls=np.array([s.fouls for s in ps], float),
            minutes=np.array([s.minutes for s in ps], float), on_court=np.array([s.on_court for s in ps]),
            off_for=np.array([0.0 if s.on_court or s.last_out is None else seen - s.last_out for s in ps]),
            margin=np.array([margin_home if s.team == g.home else -margin_home for s in ps], float),
            rate_pre=pre_g.loc[ids, "rate_pre"].to_numpy(float), min_pre=pre_g.loc[ids, "min_pre"].to_numpy(float),
            min_sd=pre_g.loc[ids, "min_sd"].to_numpy(float), pf_rate=pre_g.loc[ids, "pf_rate"].to_numpy(float),
            elapsed=seen, period=min(int(seen // 12) + 1, 4) if seen < 48 else 5,
            period_left=(12 - seen % 12) if seen < 48 else 0.0)
        final = simulate(x, p, n_sims, rng)
        frac_left = max(48 - seen, 0) / 48
        for i, pid in enumerate(ids):
            dist = PointsDistribution(pre_g.loc[pid, "pmf"])
            naive_mean = x.pts[i] + x.rate_pre[i] * x.min_pre[i] * frac_left
            for line in LADDER:
                if x.pts[i] > line:          # already decided: YES is certain
                    continue
                naive_p = 1 - stats_poisson_cdf(line - x.pts[i], max(naive_mean - x.pts[i], 1e-6) / 2.2)
                p_pre = dist.p_over(line)
                if not 0.05 <= p_pre <= 0.95:      # Kalshi lists rungs near the projection: skip far-off lines
                    continue
                kal = kalshi_mid(pid, line, tau) if kalshi_mid else None
                rows.append({"game_id": g.game_id, "player_id": pid, "T": T, "tau": tau, "seen": seen,
                             "p_kalshi": kal,
                             "quarter": min(int((T - 1e-9) // 12) + 1, 4), "line": line,
                             "pts_now": x.pts[i], "p_model": float((final[i] > line).mean()),
                             "p_pregame": p_pre, "p_naive": float(naive_p),
                             "y": float(g.final_pts[pid] > line), "low_conf": bool(pre_g.loc[pid, "low_conf"])})
    return rows


def stats_poisson_cdf(gap: float, mean_baskets: float) -> float:
    """Naive baseline: points left ≈ 2.2 × Poisson(baskets); P(points left ≤ gap)."""
    from scipy import stats
    return float(stats.poisson.cdf(np.floor(gap / 2.2), mean_baskets))


def score_table(df: pd.DataFrame, cols=("p_model", "p_pregame", "p_naive", "p_kalshi"),
                by: str = "quarter") -> pd.DataFrame:
    out = []
    for key, d in [("ALL", df)] + list(df.groupby(by)):
        y = d["y"].to_numpy()
        row = {by: key, "n": len(d), "hit_rate": float(y.mean())}
        for c in cols:
            if c in d:
                row[f"brier_{c[2:]}"] = brier(d[c].to_numpy(), y)
                row[f"logloss_{c[2:]}"] = log_loss(d[c].to_numpy(), y)
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- run / tune

LIVE_GRID = {
    "live_rate_prior_min": [50, 100, 150, 300, 600],
    "live_min_prior": [0.25, 0.5, 1.0, 2.0],
    "live_blowout_drop": [0.0, 0.3, 0.6, 0.9],
    "live_blowout_gain": [0.0, 0.5, 1.0],
    "foul_sit_prob": [0.0, 0.4, 0.7, 1.0],
    "live_min_sd_scale": [0.6, 0.8, 1.0, 1.3],
    "exit_max_prob": [0.0, 0.4, 0.8],
}


def game_ids(session: Session, season: str, start=None, end=None) -> list[str]:
    q = select(m.Game.id).where(m.Game.season == season, m.Game.status == "post",
                                m.Game.id.in_(select(m.Play.game_id).distinct()))
    if start:
        q = q.where(m.Game.game_date >= start)
    if end:
        q = q.where(m.Game.game_date <= end)
    return sorted(session.scalars(q))


def run_games(games: list[GameData], pre: pd.DataFrame, p: dict, n_sims: int, seed: int = 7,
              kalshi_mid=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in games:
        rows += snapshot_rows(g, pre, p, n_sims, rng, p["assumed_feed_delay_s"], p["sample_points"], kalshi_mid)
    return pd.DataFrame(rows)


class CandleMids:
    """Kalshi mid at any moment of a past game, from 1-minute bid/ask candles (last full minute ≤ t)."""

    def __init__(self, session: Session, client, cfg: dict, cache_dir):
        from propedge.backtest.repricing import candle_mids
        self._fetch = candle_mids
        self.client, self.cfg, self.cache_dir = client, cfg, cache_dir
        rows = session.execute(select(m.PropMarket.ticker, m.PropMarket.player_id, m.PropMarket.line,
                                      m.PropMarket.game_id)).all()
        self.ticker = {(str(pid), float(line), gid): t for t, pid, line, gid in rows if pid is not None}
        self.series: dict[str, pd.DataFrame] = {}

    def for_game(self, g: GameData):
        start = min(p.wallclock for p in g.plays if p.wallclock) - timedelta(minutes=5)
        end = max(p.wallclock for p in g.plays if p.wallclock) + timedelta(minutes=5)
        base, series = self.cfg["odds"]["kalshi"]["base_url"], self.cfg["odds"]["kalshi"]["series"]["pts"]

        def mid(pid, line, tau):
            t = self.ticker.get((pid, float(line), g.game_id))
            if t is None:
                return None
            if t not in self.series:
                self.series[t] = self._fetch(self.client, base, series, t, start, end, self.cache_dir)
            s = self.series[t]
            s = s[s["t"] <= tau] if len(s) else s
            return float(s["p"].iloc[-1]) if len(s) else None
        return mid


def tune_live(games: list[GameData], pre: pd.DataFrame, p: dict, n_sims: int = 800, rounds: int = 1) -> dict:
    """Coordinate descent on log loss over open ladder lines (same random numbers for every candidate)."""
    p = dict(p)
    for _ in range(rounds):
        for name, grid in LIVE_GRID.items():
            scores = {}
            for v in grid:
                d = run_games(games, pre, dict(p, **{name: v}), n_sims)
                scores[v] = log_loss(d["p_model"].to_numpy(), d["y"].to_numpy())
            p[name] = min(scores, key=scores.get)
            log.info("tune %s -> %s (log loss %.4f)", name, p[name], scores[p[name]])
    return p
