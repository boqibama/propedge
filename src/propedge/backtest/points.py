"""Points-model backtest: tune on one season, test on the next, with no look-ahead.

- Features come from models/points_features.py, which only uses games before each game.
- Parameters are tuned on the TUNE season and then frozen; the TEST season is scored once.
- Baselines the model must beat: season-to-date average (last season's average before a
  player's first game) and the last-10-games average.
- Calibration is checked at Kalshi-style ladder lines (10+, 15+, ...).
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.db import models as m
from dataclasses import replace

from propedge.models.bias import role_tier, rolling_bias_factors
from propedge.models.points_dist import expected_minutes, points_pmf, prob_at_least
from propedge.models.points_features import build_features
from propedge.models.points_model import project, subset

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- data

def load_frames(session: Session, seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pl = pd.read_sql(select(m.PlayerGameLog.player_id, m.PlayerGameLog.game_id, m.PlayerGameLog.season,
                            m.PlayerGameLog.game_date, m.PlayerGameLog.team, m.PlayerGameLog.opponent,
                            m.PlayerGameLog.is_home, m.PlayerGameLog.minutes, m.PlayerGameLog.pts)
                     .where(m.PlayerGameLog.season.in_(seasons)), session.get_bind())
    tm = pd.read_sql(select(m.TeamGameLog.game_id, m.TeamGameLog.season, m.TeamGameLog.game_date,
                            m.TeamGameLog.team, m.TeamGameLog.opponent, m.TeamGameLog.is_home,
                            m.TeamGameLog.minutes, m.TeamGameLog.pts, m.TeamGameLog.fga, m.TeamGameLog.fta,
                            m.TeamGameLog.oreb, m.TeamGameLog.tov)
                     .where(m.TeamGameLog.season.in_(seasons)), session.get_bind())
    gm = pd.read_sql(select(m.Game.id.label("game_id"), m.Game.home_spread, m.Game.total), session.get_bind())
    for df in (pl, tm):
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    pl = pl[pl["minutes"] > 0]          # model players who play (see report: DNP risk is separate)
    return pl, tm, gm


# ----------------------------------------------------------------------------- scoring

@dataclass
class Scores:
    n: int
    mae: float
    rmse: float
    bias: float

    @staticmethod
    def of(pred: np.ndarray, actual: np.ndarray) -> "Scores":
        err = pred - actual
        return Scores(len(err), float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), float(np.mean(err)))


def eval_rows(f: pd.DataFrame, season: str, start: date | None = None) -> np.ndarray:
    """Rows scored: the season's played games where the model AND both baselines have history."""
    mask = (f["season"] == season) & f["pts"].notna() & f["ewm_min"].notna() & \
           f["baseline_season_avg"].notna() & f["baseline_last10"].notna()
    if start is not None:
        mask &= f["game_date"] >= start
    return mask.to_numpy()


def played_rows(f: pd.DataFrame, season: str) -> np.ndarray:
    """Every played player-game of the season we can project — whole team-games (team constraints)."""
    return ((f["season"] == season) & f["pts"].notna() & f["ewm_min"].notna()).to_numpy()


def mean_prediction(f: pd.DataFrame, params: dict, keep: np.ndarray | None = None) -> tuple[np.ndarray, object]:
    """Project all rows of `f` (whole team-games), return the rows in `keep`."""
    proj = project(f, params)
    if keep is not None:
        proj = subset(proj, keep)
    return proj.rate * proj.expected_minutes, proj


def minutes_nll(f: pd.DataFrame, params: dict, keep: np.ndarray | None = None) -> float:
    """How well the minutes distribution fits actual minutes (negative log-likelihood per game)."""
    from propedge.models.points_dist import minutes_weights
    proj = project(f, params)
    if keep is not None:
        proj, f = subset(proj, keep), f[keep]
    w = minutes_weights(proj.min_mu, proj.min_sd, proj.p_exit)
    k = np.clip(np.rint(f["minutes"].to_numpy(float)).astype(int), 1, 48) - 1
    return float(-np.mean(np.log(np.maximum(w[np.arange(len(k)), k], 1e-9))))


def minutes_tail_check(f: pd.DataFrame, params: dict, keep: np.ndarray | None = None,
                       min_projection: float = 24) -> list[dict]:
    """Short-night check for confident rotation players: model vs actual P(minutes ≤ x% of projection)."""
    from propedge.models.points_dist import MINUTE_GRID, minutes_weights
    proj = project(f, params)
    if keep is not None:
        proj, f = subset(proj, keep), f[keep]
    w = minutes_weights(proj.min_mu, proj.min_sd, proj.p_exit)
    emin = w @ MINUTE_GRID
    sel = (emin >= min_projection) & ~proj.low_confidence
    act = f["minutes"].to_numpy(float)
    out = []
    for frac in (0.25, 0.5, 0.7, 0.9):
        thr = frac * emin
        p_model = (w * (MINUTE_GRID[None, :] <= thr[:, None])).sum(1)
        out.append({"share_of_projection": frac, "n": int(sel.sum()), "model": float(p_model[sel].mean()),
                    "actual": float((act[sel] <= thr[sel]).mean())})
    return out


def nll(pmf: np.ndarray, actual: np.ndarray) -> float:
    k = np.clip(actual.astype(int), 0, pmf.shape[1] - 1)
    return float(-np.mean(np.log(np.maximum(pmf[np.arange(len(k)), k], 1e-12))))


def calibration(p_model: np.ndarray, hit: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_model, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if sel.sum():
            rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n": int(sel.sum()),
                         "predicted": float(p_model[sel].mean()), "actual": float(hit[sel].mean())})
    return pd.DataFrame(rows)


def empirical_hit_rate(f: pd.DataFrame, threshold: int, window: int = 20) -> np.ndarray:
    """Baseline probability: how often the player reached the threshold in his last `window`
    games (smoothed: (hits + 0.5) / (games + 1)). No look-ahead (shifted)."""
    hit = (f["pts"] >= threshold).astype(float).where(f["pts"].notna())
    g = hit.groupby(f["player_id"])
    hits = g.transform(lambda s: s.rolling(window, min_periods=0).sum().shift(1)).fillna(0)
    n = g.transform(lambda s: s.notna().astype(float).rolling(window, min_periods=0).sum().shift(1)).fillna(0)
    return ((hits + 0.5) / (n + 1)).to_numpy()


# ----------------------------------------------------------------------------- tuning

MEAN_GRID = {   # coordinate-descent candidates for parameters that move the mean
    "shrink_minutes": [10, 25, 50, 100, 200, 400],
    "opp_def_power": [0.0, 0.5, 1.0, 1.5, 2.0],
    "pace_power": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "home_rate_bump": [0.0, 0.01, 0.02, 0.03],
    "b2b_minutes_drop": [0.0, 0.02, 0.04, 0.06],
    "blowout_starter_drop": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "blowout_bench_gain": [0.0, 0.1, 0.2, 0.4],
    "missing_minutes_bump": [0.0, 0.1, 0.2, 0.4, 0.6],
    "missing_rate_bump": [0.0, 0.2, 0.4, 0.6, 0.8],
}
HALFLIFE_GRID = {"min_halflife_games": [1.5, 2, 3, 5, 8], "rate_halflife_games": [8, 15, 25, 40, 60]}
BIAS_GRID = {"bias_prior_games": [100, 300, 1000, 3000], "bias_cap": [0.02, 0.04, 0.06, 0.08],
             "bias_window_days": [14, 30, 60, 400]}

MINUTES_GRID = {   # tuned on minutes themselves (minutes log-likelihood)
    "min_center_bump": [0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
    "min_sd_scale": [0.4, 0.5, 0.6, 0.8, 1.0, 1.2],
    "min_sd_floor": [1.0, 1.5, 2.0, 3.0, 4.0],
    "early_exit_prob": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
}

DIST_GRID = {
    "phi": [2.0, 2.4, 2.8, 3.2, 3.6, 4.2],
    "phi_power": [-0.6, -0.45, -0.3, -0.15, 0.0, 0.15],
    "rate_cv": [0.0, 0.05, 0.1, 0.2],
    "rate_cv_floor": [0.0, 0.03, 0.06],
    "low_conf_widen": [1.0, 1.25, 1.5, 2.0],
}


def fit_role_baseline(f: pd.DataFrame, season: str, min_minutes: float = 300) -> tuple[float, float]:
    """Line through season points-per-minute vs minutes-per-game (minutes-weighted)."""
    s = f[(f["season"] == season) & f["pts"].notna()].groupby("player_id").agg(
        mins=("minutes", "sum"), pts=("pts", "sum"), games=("minutes", "size"))
    s = s[s["mins"] >= min_minutes]
    mpg, rate, w = s["mins"] / s["games"], s["pts"] / s["mins"], s["mins"]
    slope, intercept = np.polyfit(mpg, rate, 1, w=np.sqrt(w))
    return float(intercept), float(slope)


def tune(players, teams, games, params: dict, season: str, start: date, dist_sample: int = 6000,
         rounds: int = 2) -> dict:
    p = dict(params)
    # The mean is tuned on squared error (RMSE), not MAE: MAE rewards hitting the MEDIAN, and since
    # points are right-skewed that drags projections low — bad for probabilities at ladder lines.
    # 1. Half-lives need the features rebuilt: small grid.
    best = None
    feats = {}
    for hm, hr in itertools.product(HALFLIFE_GRID["min_halflife_games"], HALFLIFE_GRID["rate_halflife_games"]):
        q = dict(p, min_halflife_games=hm, rate_halflife_games=hr)
        f = build_features(players, teams, games, q)
        P = played_rows(f, season)
        sc = eval_rows(f, season, start)[P]
        rmse = Scores.of(mean_prediction(f[P], q, sc)[0], f[P]["pts"].to_numpy()[sc]).rmse
        log.info("tune halflives min=%s rate=%s RMSE=%.4f", hm, hr, rmse)
        feats[(hm, hr)] = f
        if best is None or rmse < best[0]:
            best = (rmse, hm, hr)
    _, p["min_halflife_games"], p["rate_halflife_games"] = best
    f = feats[(best[1], best[2])]
    p["baseline_rate_intercept"], p["baseline_rate_slope"] = fit_role_baseline(f, season)
    P = played_rows(f, season)
    fp = f[P]                                   # whole team-games (projected together)…
    sc = eval_rows(f, season, start)[P]         # …scored on these rows
    actual = fp["pts"].to_numpy()[sc]

    # 2a. Minutes distribution: fit to minutes directly (all tune rows: it's cheap).
    for _ in range(rounds):
        for name, grid in MINUTES_GRID.items():
            scores = {v: minutes_nll(fp, dict(p, **{name: v}), sc) for v in grid}
            p[name] = min(scores, key=scores.get)
            log.info("tune %s -> %s (minutes NLL %.4f)", name, p[name], scores[p[name]])

    # 2. Mean parameters: coordinate descent on RMSE.
    for _ in range(rounds):
        for name, grid in MEAN_GRID.items():
            scores = {v: Scores.of(mean_prediction(fp, dict(p, **{name: v}), sc)[0], actual).rmse for v in grid}
            p[name] = min(scores, key=scores.get)
            log.info("tune %s -> %s (RMSE %.4f)", name, p[name], scores[p[name]])

    # 3. Rolling bias correction: settings chosen on the tune season, using each day's earlier results.
    season_rows = ((f["season"] == season) & f["pts"].notna() & f["ewm_min"].notna()).to_numpy()
    fs_all = f[season_rows]
    base = mean_prediction(fs_all, p)[0]
    scored = (fs_all["game_date"] >= start).to_numpy() & eval_rows(f, season, start)[season_rows]
    best = None
    for k, cap, win in itertools.product(*BIAS_GRID.values()):
        fac = rolling_bias_factors(fs_all["season"], fs_all["game_date"], base, fs_all["pts"].to_numpy(), k, cap, win)
        rmse = Scores.of((base * fac)[scored], fs_all["pts"].to_numpy()[scored]).rmse
        if best is None or rmse < best[0]:
            best = (rmse, k, cap, win)
    _, p["bias_prior_games"], p["bias_cap"], p["bias_window_days"] = best
    log.info("tune bias correction -> prior %s, cap %s, window %s days (RMSE %.4f, without %.4f)", best[1], best[2],
             best[3], best[0], Scores.of(base[scored], fs_all["pts"].to_numpy()[scored]).rmse)

    # 4. Points spread parameters: coordinate descent on negative log-likelihood (a sample, for speed).
    rng = np.random.default_rng(0)
    scored_idx = np.flatnonzero(sc)
    sample = np.sort(rng.choice(scored_idx, size=min(dist_sample, len(scored_idx)), replace=False))
    act_s = fp["pts"].to_numpy()[sample]
    for _ in range(rounds):
        for name, grid in DIST_GRID.items():
            scores = {v: nll(subset(project(fp, dict(p, **{name: v})), sample).pmf(), act_s) for v in grid}
            p[name] = min(scores, key=scores.get)
            log.info("tune %s -> %s (NLL %.4f)", name, p[name], scores[p[name]])
    return p


# ----------------------------------------------------------------------------- evaluation

def evaluate(players, teams, games, params: dict, season: str, lines: list[int]) -> dict:
    f = build_features(players, teams, games, params)
    # Bias factors come from ALL of this season's earlier played games (not just the scored rows).
    season_rows = ((f["season"] == season) & f["pts"].notna() & f["ewm_min"].notna()).to_numpy()
    fs_all = f[season_rows]
    base_all, proj_all = mean_prediction(fs_all, params)
    tiers = role_tier(proj_all.expected_minutes, params["bias_tier_edges"]) if params["bias_by_tier"] else None
    fac_all = rolling_bias_factors(fs_all["season"], fs_all["game_date"], base_all, fs_all["pts"].to_numpy(),
                                   params["bias_prior_games"], params["bias_cap"], params["bias_window_days"], tiers)
    sc = eval_rows(f, season)[season_rows]
    fr = fs_all[sc].reset_index(drop=True)
    factor = fac_all[sc]
    actual = fr["pts"].to_numpy()
    pred_raw, proj = mean_prediction(fs_all, params, sc)
    proj = replace(proj, rate=proj.rate * factor)
    pred = pred_raw * factor
    pmf = proj.pmf()
    out = {"season": season, "rows": int(len(fr)),
           "bias_factor": {"mean": float(factor.mean()), "min": float(factor.min()), "max": float(factor.max())}}

    def block(mask):
        return {"model": Scores.of(pred[mask], actual[mask]).__dict__,
                "model_no_bias_correction": Scores.of(pred_raw[mask], actual[mask]).__dict__,
                "season_avg": Scores.of(fr["baseline_season_avg"].to_numpy()[mask], actual[mask]).__dict__,
                "last10": Scores.of(fr["baseline_last10"].to_numpy()[mask], actual[mask]).__dict__}

    low = proj.low_confidence
    early = (fr["season_games"] < 10).to_numpy()
    spread_rows = (fr["margin_source"] == "spread").to_numpy()
    out["all"] = block(np.ones(len(fr), bool))
    out["normal_confidence"] = block(~low)
    out["low_confidence"] = block(low)
    out["first_10_games_of_season"] = block(early)
    out["first_10_games_normal_confidence"] = block(early & ~low)
    out["rest_of_season"] = block(~early)
    out["low_conf_reasons"] = pd.Series([r for r in proj.low_reasons if r]).str.split(",").explode() \
        .value_counts().to_dict()
    out["share_with_real_spread"] = float(spread_rows.mean())
    out["nll"] = nll(pmf, actual)
    out["minutes_tail"] = minutes_tail_check(fs_all, params, sc)

    cal = {}
    for line in lines:
        p_model = prob_at_least(pmf, line)
        hit = (actual >= line).astype(float)
        p_base = empirical_hit_rate(f, line)[season_rows][sc]
        cal[line] = {
            "base_rate": float(hit.mean()),
            "brier_model": float(np.mean((p_model - hit) ** 2)),
            "brier_last20_hit_rate": float(np.mean((p_base - hit) ** 2)),
            "brier_model_normal_conf": float(np.mean((p_model[~low] - hit[~low]) ** 2)),
            "brier_last20_normal_conf": float(np.mean((p_base[~low] - hit[~low]) ** 2)),
            "table": calibration(p_model[~low], hit[~low]).to_dict("records"),
        }
    out["calibration"] = cal
    return out


def write_tables(result: dict, tuned: dict, path: Path) -> None:
    path.write_text(json.dumps({"result": result, "params": tuned}, indent=2, default=float))
