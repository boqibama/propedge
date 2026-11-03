"""Pre-game points projection. Pure math on the feature table, no I/O.

For every player-game it produces the inputs of the points distribution (points_dist.py):

MINUTES
    base      = recent minutes (exponentially weighted: recent games count more)
    × back-to-back factor           (tired legs / rest management)
    × blowout factor                (starters sit when a game gets out of hand; bench players gain)
    × teammates-out factor          (more minutes when rotation players are newly out)
    sd        = the player's own recent minutes spread, with a floor, widened if low-confidence

TEAM CONSTRAINTS
    a team-game's projected minutes are scaled down to fit 240, and its projected points to fit the
    team's expected score (from the betting total/spread) — never scaled up

RATE (points per minute)
    observed  = recent points ÷ recent minutes (weighted)
    shrunk toward a role baseline (league average rate for players with similar minutes) —
    a lot when we have few minutes of evidence, barely when we have a full season
    × opponent defence factor ^ γ   (points allowed per 100 possessions vs league)
    × pace factor ^ δ               (more possessions -> more points per minute)
    × home factor, × teammates-out usage bump

CONFIDENCE
    "low" for rookies, players new to their team, players back from an absence, players with
    too few minutes in our data, and teammates of big new arrivals early in the season. Their uncertainty is widened and v1 never flags them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from propedge.models.points_dist import expected_minutes, points_pmf


@dataclass(frozen=True)
class Projection:
    min_mu: np.ndarray
    min_sd: np.ndarray
    rate: np.ndarray
    rate_cv: np.ndarray
    phi: float
    phi_power: float
    p_exit: float
    low_confidence: np.ndarray       # bool
    low_reasons: list[str]           # per row, comma-separated ("" if confident)

    @property
    def expected_minutes(self) -> np.ndarray:
        return expected_minutes(self.min_mu, self.min_sd, self.p_exit)

    @property
    def mean_points(self) -> np.ndarray:
        return self.rate * self.expected_minutes

    def pmf(self, chunk: int = 2000) -> np.ndarray:
        return points_pmf(self.rate, self.min_mu, self.min_sd, self.phi, self.rate_cv, self.phi_power, self.p_exit,
                          chunk=chunk)


def role_baseline_rate(min_base: np.ndarray, p: dict) -> np.ndarray:
    """League-typical points per minute for a player who plays `min_base` minutes (fitted line)."""
    return np.clip(p["baseline_rate_intercept"] + p["baseline_rate_slope"] * min_base, 0.15, 1.0)


def blowout_probability(exp_margin: np.ndarray, p: dict) -> np.ndarray:
    """P(final margin beyond ±blowout_margin) if margin ~ Normal(expected margin, margin_sd)."""
    m, sd, b = np.nan_to_num(exp_margin), p["margin_sd"], p["blowout_margin"]
    return stats.norm.sf((b - m) / sd) + stats.norm.cdf((-b - m) / sd)


def project(f: pd.DataFrame, p: dict) -> Projection:
    """Projection for every row of the feature table `f` with parameters `p` (config model.points)."""
    min_base = f["ewm_min"].to_numpy(float)
    b2b = (f["days_rest"] == 1).to_numpy()
    blow = blowout_probability(f["exp_margin"].to_numpy(float), p)
    starter_w = np.clip((min_base - p["bench_minutes"]) / (p["starter_minutes"] - p["bench_minutes"]), 0, 1)
    missing = f["missing_share"].to_numpy(float)

    min_mu = (min_base
              * np.where(b2b, 1 - p["b2b_minutes_drop"], 1.0)
              * (1 - p["blowout_starter_drop"] * blow * starter_w)
              * (1 + p["blowout_bench_gain"] * blow * (1 - starter_w))
              * (1 + p["missing_minutes_bump"] * missing))
    # The recent-minutes average includes short nights (injury exits, blowouts) that the early-exit
    # component of the minutes distribution models separately; the "normal night" centre sits higher.
    min_mu = np.clip(min_mu * (1 + p["min_center_bump"]), 1, p["max_minutes"])

    # Rate: shrink the observed rate toward the role baseline by evidence (minutes seen).
    evidence = np.minimum(f["career_min"].to_numpy(float), p["evidence_cap_minutes"])
    rate_obs = f["rate_obs"].to_numpy(float)
    base_rate = role_baseline_rate(min_base, p)
    rate_obs = np.where(np.isnan(rate_obs), base_rate, rate_obs)
    k = p["shrink_minutes"]
    rate = (evidence * rate_obs + k * base_rate) / (evidence + k)
    rate *= np.nan_to_num(f["opp_def_factor"].to_numpy(float), nan=1.0) ** p["opp_def_power"]
    rate *= np.nan_to_num(f["pace_factor"].to_numpy(float), nan=1.0) ** p["pace_power"]
    rate *= np.where(f["is_home"].to_numpy(bool), 1 + p["home_rate_bump"], 1.0)
    rate *= 1 + p["missing_rate_bump"] * missing

    # Team constraints (Phase 3 diagnosis): early in a season every player carries last season's role, and a
    # team's projections added up to far more than 240 minutes and its expected score. Scale DOWN (never up)
    # so a team-game's projected minutes fit in regulation and its points fit its expected total.
    if p.get("team_constraints", True) and "team_exp_pts" in f:
        key = f["team"].astype(str) + "|" + f["game_id"].astype(str)
        min_sum = pd.Series(min_mu, index=f.index).groupby(key).transform("sum").to_numpy()
        excess = np.maximum(min_sum - p["team_minutes_cap"], 0.0)
        if p.get("team_minutes_method", "variance") == "proportional":
            min_mu = min_mu * np.minimum(1.0, p["team_minutes_cap"] / np.maximum(min_sum, 1e-9))
        else:
            # Take the excess from where minutes are least certain: share ∝ the player's minutes variance
            # (≈ equal minutes off everyone: a big slice of a bench role, a small slice of a star's).
            sd = np.nan_to_num(f["ewm_min_sd"].to_numpy(float), nan=p["min_sd_floor"] * 2)
            var = sd ** 2 * p["min_sd_scale"] ** 2 + p["min_sd_floor"] ** 2
            var_sum = pd.Series(var, index=f.index).groupby(key).transform("sum").to_numpy()
            min_mu = np.maximum(min_mu - excess * var / np.maximum(var_sum, 1e-9), 1.0)
        pts_sum = pd.Series(rate * min_mu, index=f.index).groupby(key).transform("sum").to_numpy()
        target = f["team_exp_pts"].to_numpy(float) * p["team_points_share"]
        pts_scale = np.where(np.isfinite(target), np.minimum(1.0, target / np.maximum(pts_sum, 1e-9)), 1.0)
        rate = rate * pts_scale

    # Confidence.
    flags = {
        "rookie": f["rookie"].to_numpy(bool),
        "new_team": f["new_team"].to_numpy(bool),
        "returning": f["returning"].to_numpy(bool),
        "small_sample": f["career_min"].to_numpy(float) < p["min_minutes_sample"],
        "no_history": f["history_unknown"].to_numpy(bool),
        # Teammate of significant new arrivals, during the team's first games together.
        "teammates_new": ((f["team_new_share"].to_numpy(float) >= p["team_new_share_threshold"])
                          & (f["games_before"].to_numpy(float) < p["new_team_games"])
                          & ~f["new_team"].to_numpy(bool)),
    }
    low = np.zeros(len(f), bool)
    for v in flags.values():
        low |= v
    reasons = [",".join(name for name, v in flags.items() if v[i]) for i in range(len(f))]

    sd_hist = np.nan_to_num(f["ewm_min_sd"].to_numpy(float), nan=p["min_sd_floor"] * 2)
    min_sd = np.sqrt(sd_hist ** 2 * p["min_sd_scale"] ** 2 + p["min_sd_floor"] ** 2)
    rate_cv = p["rate_cv"] * np.sqrt(k / (evidence + k)) + p["rate_cv_floor"]
    widen = np.where(low, p["low_conf_widen"], 1.0)
    return Projection(min_mu=min_mu, min_sd=min_sd * widen, rate=rate, rate_cv=rate_cv * widen,
                      phi=p["phi"], phi_power=p["phi_power"], p_exit=p["early_exit_prob"],
                      low_confidence=low, low_reasons=reasons)


def subset(proj: Projection, keep: np.ndarray) -> Projection:
    """Rows `keep` (bool mask or indices) of a projection. Project whole team-games first (the team
    constraints need every player of the game), then score only the rows you want."""
    return Projection(min_mu=proj.min_mu[keep], min_sd=proj.min_sd[keep], rate=proj.rate[keep],
                      rate_cv=proj.rate_cv[keep], phi=proj.phi, phi_power=proj.phi_power, p_exit=proj.p_exit,
                      low_confidence=proj.low_confidence[keep],
                      low_reasons=list(np.asarray(proj.low_reasons, dtype=object)[keep]))
