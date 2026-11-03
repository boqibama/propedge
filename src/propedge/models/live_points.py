"""Live points engine: final-points distribution for players in a game in progress. Pure, no I/O.

final points = points so far + points in the remaining minutes, simulated `n_sims` times.

REMAINING MINUTES (per simulation)
  1. Planned total minutes: the pre-game projection, blended toward his pace so far as the game goes on
     (by halftime the two count about equally; `live_min_prior` sets how fast).
  2. Blowout meter (0–1): P(final margin ≥ blowout margin) given the score and time left. Starters lose
     minutes and bench players gain them in proportion to it (garbage time).
  3. Foul trouble: at the coach's usual limits (2 fouls in Q1, 3 in Q2, 4 in Q3, 5 in Q4) he likely sits
     the rest of the period. Future fouls are simulated; a 6th foul ends his night at that moment.
  4. Exit detection: a starter who has been off the floor far longer than a normal rest (and it isn't
     garbage time) has probably left the game (injury/illness) — then he plays 0 more minutes.
  5. Future early exits (injury) and a small chance of overtime.
  6. Noise around the plan, cut to [0, minutes left].

RATE FOR THE REST OF THE GAME (points per minute)
  The pre-game rate counts as `live_rate_prior_min` minutes of evidence; tonight's points/minutes are
  added to it. A big prior = we don't chase a hot hand (in-game streaks predict little).

POINTS IN THE REMAINING MINUTES: negative binomial with the same spread rule as pre-game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

FOUL_OUT = 6


@dataclass
class LiveInputs:
    """Arrays, one entry per player (all in the same game)."""
    pts: np.ndarray            # points so far
    fouls: np.ndarray          # personal fouls so far
    minutes: np.ndarray        # minutes played so far
    on_court: np.ndarray       # bool
    off_for: np.ndarray        # game minutes since he last checked out (0 if on the floor)
    margin: np.ndarray         # his team's lead (negative if trailing)
    rate_pre: np.ndarray       # pre-game points per minute (incl. bias correction)
    min_pre: np.ndarray        # pre-game normal-night minutes (centre of the minutes distribution)
    min_sd: np.ndarray         # pre-game minutes sd
    pf_rate: np.ndarray        # personal fouls per minute (his history)
    elapsed: float             # game minutes elapsed
    period: int
    period_left: float         # minutes left in the current period


def starter_weight(min_pre: np.ndarray, p: dict) -> np.ndarray:
    return np.clip((min_pre - p["bench_minutes"]) / (p["starter_minutes"] - p["bench_minutes"]), 0, 1)


def blowout_meter(margin: np.ndarray, minutes_left: float, p: dict) -> np.ndarray:
    """P(|final margin| ≥ blowout margin): the lead drifts like a random walk over the time left."""
    sd = max(p["margin_sd"] * np.sqrt(max(minutes_left, 0.0) / 48), 1e-6)
    b = p["live_blowout_margin"]
    m = np.abs(margin)
    return stats.norm.sf((b - m) / sd) + stats.norm.cdf((-b - m) / sd)


def foul_trouble(fouls: np.ndarray, period: int, p: dict) -> np.ndarray:
    limits = p["foul_limits"]                       # e.g. [2, 3, 4, 5] for Q1..Q4
    limit = limits[min(period, 4) - 1]
    return fouls >= limit


def exit_probability(x: LiveInputs, meter: np.ndarray, p: dict) -> np.ndarray:
    """Chance he has already left for good: a starter off the floor much longer than a normal rest."""
    sw = starter_weight(x.min_pre, p)
    over = np.clip((x.off_for - p["exit_after_off_min"]) / p["exit_ramp_min"], 0, 1)
    return np.where(~x.on_court & (sw > 0.5), over * p["exit_max_prob"] * (1 - meter), 0.0)


def simulate(x: LiveInputs, p: dict, n_sims: int, rng: np.random.Generator,
             return_parts: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    """(players, n_sims) array of simulated FINAL points."""
    n = len(x.pts)
    in_regulation = x.elapsed < 48
    left = max(48 - x.elapsed, 0.0) if in_regulation else max(x.period_left, 0.0)
    frac = min(x.elapsed / 48, 1.0)

    # 1. planned total minutes: pre-game plan blended with tonight's pace
    w = frac / (frac + p["live_min_prior"])
    pace_total = np.where(frac > 0, x.minutes / max(frac, 1e-6), x.min_pre)
    planned_total = (1 - w) * x.min_pre + w * pace_total
    planned_left = np.clip(planned_total - x.minutes, 0, left)
    share = np.where(left > 0, planned_left / max(left, 1e-9), 0.0)

    # 2. blowout
    meter = blowout_meter(x.margin, left, p)
    sw = starter_weight(x.min_pre, p)
    planned_left = planned_left * (1 - p["live_blowout_drop"] * meter * sw) \
        * (1 + p["live_blowout_gain"] * meter * (1 - sw))
    planned_left = np.clip(planned_left, 0, left)

    # 3a. foul trouble now: likely sits the rest of this period
    trouble = foul_trouble(x.fouls, x.period, p) & in_regulation
    sits = rng.random((n, n_sims)) < (trouble * p["foul_sit_prob"])[:, None]
    m_left = np.broadcast_to(planned_left[:, None], (n, n_sims)).copy()
    m_left -= sits * (share * min(x.period_left, left))[:, None]

    # 6. noise around the plan
    sd = x.min_sd * np.sqrt(left / 48) * p["live_min_sd_scale"]
    m_left += rng.normal(0, 1, (n, n_sims)) * sd[:, None]
    m_left = np.clip(m_left, 0, left)

    # 4. already left the game?
    p_gone = exit_probability(x, meter, p)
    m_left *= rng.random((n, n_sims)) >= p_gone[:, None]

    # 5. future early exit (injury), and overtime
    exit_now = rng.random((n, n_sims)) < p["early_exit_prob"] * (left / 48)
    m_left = np.where(exit_now, m_left * rng.random((n, n_sims)), m_left)
    if in_regulation:
        final_margin_sd = max(p["margin_sd"] * np.sqrt(left / 48), 1e-6)
        p_ot = stats.norm.cdf((0.5 - np.abs(x.margin)) / final_margin_sd) - \
            stats.norm.cdf((-0.5 - np.abs(x.margin)) / final_margin_sd)
        ot = rng.random((n, n_sims)) < p_ot[:, None]
        m_left += ot * (5.0 * np.clip(planned_total / 40, 0, 1))[:, None]

    # 3b. a 6th foul ends his night when it happens
    fouls_left = np.maximum(FOUL_OUT - x.fouls, 0)
    k = rng.poisson((x.pf_rate[:, None] * m_left))
    fouled_out = k >= fouls_left[:, None]
    j = np.maximum(fouls_left[:, None], 1)
    when = rng.beta(j, np.maximum(k - j + 1, 1))            # time of the (6 − fouls)-th foul, as a share
    m_left = np.where(fouled_out, m_left * when, m_left)
    m_left = np.where((x.fouls >= FOUL_OUT)[:, None], 0.0, m_left)

    # rate for the rest of the game: pre-game rate as a prior worth `live_rate_prior_min` minutes
    k_r = p["live_rate_prior_min"]
    rate = (k_r * x.rate_pre + x.pts) / (k_r + x.minutes)
    mean = rate[:, None] * m_left
    var = p["phi"] * mean * (np.maximum(mean, 0.5) / 10) ** p["phi_power"]
    var = np.maximum(var, mean * 1.0001 + 1e-9)
    nb_p = np.where(mean > 0, mean / var, 1.0)
    nb_n = np.where(mean > 0, mean ** 2 / np.maximum(var - mean, 1e-9), 1.0)
    rest = np.where(mean > 0, rng.negative_binomial(np.maximum(nb_n, 1e-9), np.clip(nb_p, 1e-9, 1)), 0)
    final = x.pts[:, None] + rest
    if return_parts:
        return final, {"blowout_meter": meter, "foul_trouble": trouble, "p_gone": p_gone,
                       "planned_left": planned_left, "rate": rate, "minutes_left_mean": m_left.mean(axis=1)}
    return final


def prob_over(final: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """P(final > line) for each player (rows) at the given line (one line per row)."""
    return (final > lines[:, None]).mean(axis=1)
