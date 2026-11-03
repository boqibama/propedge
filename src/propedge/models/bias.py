"""Rolling bias correction for the points model. Pure math, no I/O.

The backtest showed the model running ~0.3 points high in 2025-26. Rather than re-tune on the
test season (cheating), we correct with what an honest forecaster could know on each day:
this season's already-settled games.

    ratio  = (Σ actual − Σ projected) / Σ projected        over settled player-games in the window
    shrunk = ratio × n / (n + prior_games)                  little data -> little correction
    factor = 1 + clip(shrunk, −cap, +cap)                   never moves the model more than `cap`

The factor multiplies every player's scoring rate. It can be fitted separately per role tier (bench,
rotation, starter by projected minutes): early in a season the bench was over-projected far more.
Projections compared are the model's own pre-game means *without* the correction, so the correction
can't chase its own tail. It is fitted on settled RESULTS only — never on market prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bias_factor(projected: np.ndarray, actual: np.ndarray, prior_games: float, cap: float) -> float:
    projected, actual = np.asarray(projected, float), np.asarray(actual, float)
    if len(projected) == 0 or projected.sum() <= 0:
        return 1.0
    ratio = (actual.sum() - projected.sum()) / projected.sum()
    shrunk = ratio * len(projected) / (len(projected) + prior_games)
    return float(1 + np.clip(shrunk, -cap, cap))


def role_tier(projected_minutes: np.ndarray, edges: list[float]) -> np.ndarray:
    """Role tier by projected minutes, e.g. edges [18, 26] -> 0 bench, 1 rotation, 2 starter."""
    return np.digitize(np.asarray(projected_minutes, float), edges)


def rolling_bias_factors(season: pd.Series, game_date: pd.Series, projected: np.ndarray, actual: np.ndarray,
                         prior_games: float, cap: float, window_days: float,
                         group: np.ndarray | None = None) -> np.ndarray:
    """Factor for every row, using only rows of the SAME season (and same `group`, e.g. role tier) on
    EARLIER dates within `window_days`. Rows with actual = NaN (not played yet) contribute nothing."""
    if group is not None:
        out = np.ones(len(projected))
        g = np.asarray(group)
        for key in np.unique(g):
            sel = g == key
            out[sel] = rolling_bias_factors(season[sel], game_date[sel], np.asarray(projected)[sel],
                                            np.asarray(actual)[sel], prior_games, cap, window_days)
        return out
    df = pd.DataFrame({"season": season.to_numpy(), "date": pd.to_datetime(game_date).to_numpy(),
                       "proj": np.asarray(projected, float), "act": np.asarray(actual, float)})
    settled = df.dropna(subset=["act", "proj"])
    daily = settled.groupby(["season", "date"]).agg(proj=("proj", "sum"), act=("act", "sum"), n=("proj", "size"))
    out = np.ones(len(df))
    factors: dict[tuple, float] = {}
    for (s, d) in df[["season", "date"]].drop_duplicates().itertuples(index=False):
        if s not in daily.index.get_level_values(0):
            factors[(s, d)] = 1.0
            continue
        days = daily.loc[s]
        window = days[(days.index < d) & (days.index >= d - pd.Timedelta(days=window_days))]
        n, p, a = window["n"].sum(), window["proj"].sum(), window["act"].sum()
        if n == 0 or p <= 0:
            factors[(s, d)] = 1.0
        else:
            factors[(s, d)] = float(1 + np.clip((a - p) / p * n / (n + prior_games), -cap, cap))
    for i, key in enumerate(zip(df["season"], df["date"])):
        out[i] = factors[key]
    return out
