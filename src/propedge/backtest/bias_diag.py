"""Where does the pre-game over-projection come from? Minutes vs scoring rate, by role and season stage.

For a group of settled player-games:
    projected points = projected rate × projected minutes
    actual points    = actual rate    × actual minutes
so the relative error splits (approximately, in logs) into
    minutes part = log(Σ projected minutes / Σ actual minutes)
    rate part    = log(projected pts-per-min / actual pts-per-min)
A positive part means "too high". All projections are pre-game, built only from earlier games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from propedge.backtest.points import load_frames
from propedge.models.bias import role_tier, rolling_bias_factors
from propedge.models.points_features import build_features
from propedge.models.points_model import project


def projection_table(session: Session, cfg: dict, seasons: list[str], with_bias: bool = True) -> pd.DataFrame:
    params = cfg["model"]["points"]
    players, teams, games = load_frames(session, cfg["history"]["seasons"])
    f = build_features(players, teams, games, params)
    f = f[f["season"].isin(seasons) & f["ewm_min"].notna() & f["pts"].notna()].reset_index(drop=True)
    proj = project(f, params)
    emin = proj.expected_minutes
    pts = proj.rate * emin
    fac = np.ones(len(f))
    if with_bias:
        tiers = role_tier(emin, params["bias_tier_edges"]) if params["bias_by_tier"] else None
        fac = rolling_bias_factors(f["season"], f["game_date"], pts, f["pts"].to_numpy(), params["bias_prior_games"],
                                   params["bias_cap"], params["bias_window_days"], tiers)
    return pd.DataFrame({
        "season": f["season"], "game_date": f["game_date"], "player_id": f["player_id"], "team": f["team"],
        "season_games": f["season_games"], "proj_min": emin, "proj_rate": proj.rate * fac, "proj_pts": pts * fac,
        "act_min": f["minutes"], "act_pts": f["pts"], "low_conf": proj.low_confidence,
        "low_reasons": proj.low_reasons, "new_team": f["new_team"], "team_games": f["games_before"],
    })


def decompose(t: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    def one(d):
        pm, am, pp, ap = d["proj_min"].sum(), d["act_min"].sum(), d["proj_pts"].sum(), d["act_pts"].sum()
        return pd.Series({
            "n": len(d), "proj_pts": pp / len(d), "act_pts": ap / len(d), "pts_bias": (pp - ap) / len(d),
            "pts_bias_pct": 100 * (pp / ap - 1),
            "minutes_part_pct": 100 * np.log(pm / am), "rate_part_pct": 100 * np.log((pp / pm) / (ap / am)),
            "proj_min": pm / len(d), "act_min": am / len(d),
        })
    return t.groupby(by, observed=True).apply(one, include_groups=False).reset_index()


def role(t: pd.DataFrame) -> pd.Series:
    return pd.cut(t["proj_min"], [0, 18, 26, 32, 48], labels=["bench <18", "rotation 18-26", "starter 26-32",
                                                                 "star 32+"])


def stage(t: pd.DataFrame) -> pd.Series:
    return pd.cut(t["season_games"], [-1, 2, 9, 29, 200], labels=["games 0-2", "games 3-9", "games 10-29",
                                                                   "games 30+"])
