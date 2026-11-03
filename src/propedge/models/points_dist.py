"""Points distribution for one player-game. Pure math, no I/O.

Plain-language version:
1. Minutes are uncertain, so we use a *distribution* of minutes: a normal curve around the
   projected minutes, cut off at 1 and 48 (we model players who play).
2. For each possible minutes value m, points follow a negative binomial with
       mean     = rate × m
       variance = φ × mean × (mean / 10)^κ + (cv × mean)²
   - φ > 1 because points come in 2s and 3s (a Poisson count of *baskets* times ~2.2 points
     is more spread out than a Poisson count of *points*).
   - κ (usually slightly negative) lets low scorers be relatively more erratic than stars:
     a bench player's 4 points is a coin-flip; a star's 25 is steadier relative to its size.
   - (cv × mean)² adds our uncertainty about the rate itself (bigger for small samples).
3. The final distribution mixes step 2 over the minutes in step 1.

Everything is vectorised over many players at once (for backtests); `chunk` keeps memory bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

MAX_POINTS = 100
MINUTE_GRID = np.arange(1, 49, dtype=float)       # 1..48 minutes


def minutes_weights(mu: np.ndarray, sd: np.ndarray, p_exit: float = 0.0) -> np.ndarray:
    """(N, 48) probability of playing m minutes.

    Mixture of
      (1 − p_exit) × normal(mu, sd) cut to [0.5, 48.5]     a normal night, minutes close to plan
      p_exit       × uniform on 1 .. mu                      left early (injury, ejection, illness)
    Real minutes are tight around the plan with a small separate tail of early exits; one wide
    bell curve put twice too much weight on short nights (Phase 3 check)."""
    mu, sd = np.asarray(mu, float)[:, None], np.maximum(np.asarray(sd, float)[:, None], 0.5)
    edges_hi = stats.norm.cdf((MINUTE_GRID + 0.5 - mu) / sd)
    edges_lo = stats.norm.cdf((MINUTE_GRID - 0.5 - mu) / sd)
    w = np.clip(edges_hi - edges_lo, 0, None)
    total = w.sum(axis=1, keepdims=True)
    # If mu is far outside [1, 48] all bins can underflow; fall back to the nearest edge.
    empty = total[:, 0] <= 1e-12
    if empty.any():
        nearest = np.clip(np.rint(mu[empty, 0]), 1, 48).astype(int) - 1
        w[empty] = 0
        w[empty, nearest] = 1
        total[empty] = 1
    w = w / total
    if p_exit > 0:
        upto = np.clip(np.rint(mu[:, 0]), 1, 48).astype(int)
        exit_w = (MINUTE_GRID[None, :] <= upto[:, None]).astype(float)
        exit_w /= exit_w.sum(axis=1, keepdims=True)
        w = (1 - p_exit) * w + p_exit * exit_w
    return w


def nb_pmf(mean: np.ndarray, var: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Negative binomial pmf with the given mean and variance (variance must exceed the mean)."""
    var = np.maximum(var, mean * 1.0001 + 1e-9)
    p = mean / var
    n = mean ** 2 / (var - mean)
    return stats.nbinom.pmf(k, n, p)


def points_pmf(rate: np.ndarray, min_mu: np.ndarray, min_sd: np.ndarray, phi: float | np.ndarray,
               rate_cv: np.ndarray, phi_power: float = 0.0, p_exit: float = 0.0, max_points: int = MAX_POINTS,
               chunk: int = 2000) -> np.ndarray:
    """(N, max_points + 1) pmf of points for N player-games. The last cell holds the tail mass."""
    rate, min_mu, min_sd, rate_cv = (np.atleast_1d(np.asarray(x, float)) for x in (rate, min_mu, min_sd, rate_cv))
    phi = np.broadcast_to(np.asarray(phi, float), rate.shape)
    k = np.arange(max_points + 1)
    out = np.empty((len(rate), max_points + 1))
    for s in range(0, len(rate), chunk):
        sl = slice(s, s + chunk)
        w = minutes_weights(min_mu[sl], min_sd[sl], p_exit)                  # (n, M)
        mean = rate[sl, None] * MINUTE_GRID[None, :]                         # (n, M)
        var = phi[sl, None] * mean * (np.maximum(mean, 0.5) / 10) ** phi_power + (rate_cv[sl, None] * mean) ** 2
        pmf = nb_pmf(mean[..., None], var[..., None], k[None, None, :])      # (n, M, K)
        mixed = np.einsum("nm,nmk->nk", w, pmf)
        mixed[:, -1] += np.clip(1 - mixed.sum(axis=1), 0, None)              # tail beyond max_points
        out[sl] = mixed
    return out


def expected_minutes(min_mu: np.ndarray, min_sd: np.ndarray, p_exit: float = 0.0) -> np.ndarray:
    return minutes_weights(min_mu, min_sd, p_exit) @ MINUTE_GRID


def prob_at_least(pmf: np.ndarray, threshold: int | float) -> np.ndarray:
    """P(points ≥ threshold). Kalshi's "25+ points" is threshold 25 (= over 24.5)."""
    t = int(np.ceil(threshold))
    return pmf[:, t:].sum(axis=1)


@dataclass(frozen=True)
class PointsDistribution:
    """One player's projected points distribution."""
    pmf: np.ndarray            # P(points = k), k = 0..MAX_POINTS (last cell = tail)

    @property
    def mean(self) -> float:
        return float(np.arange(len(self.pmf)) @ self.pmf)

    @property
    def sd(self) -> float:
        k = np.arange(len(self.pmf))
        return float(np.sqrt(((k - self.mean) ** 2) @ self.pmf))

    def p_over(self, line: float) -> float:
        """P(points > line), e.g. line 24.5 for Kalshi's 25+."""
        return float(self.pmf[int(np.floor(line)) + 1:].sum())

    def quantile(self, q: float) -> int:
        return int(np.searchsorted(np.cumsum(self.pmf), q))
