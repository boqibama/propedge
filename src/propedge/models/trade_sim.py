"""Buy-then-sell on recorded Kalshi quotes: enter at the ask, exit later at the bid or with a resting
limit sell. Pure functions, no I/O.

Prices are for OUR side of a contract. YES trades at the YES bid/ask; NO trades at
    NO ask = 1 − YES bid,   NO bid = 1 − YES ask      (a NO bid at x is a YES ask at 1 − x)
Net P&L per contract = exit price − exit fee − (entry ask + entry fee).

Quotes are stored only when they change: each one held from `t0` (first seen) to `t1` (last confirmed).
A quote counts at time t only if it was confirmed within `max_age_s` before t: a gap in our polling
or a market that stopped trading is "no quote", never an old price.

Resting limit sell (the "maker" exit): post one cent inside the spread (at the ask when the spread is
1¢). It counts as filled only if, while it rests, a recorded best bid for our side reaches its price —
a bid at or above our offer can't coexist with it, so our offer must have been taken. Polling every
~3 s can miss a brief cross, so this undercounts fills (conservative). Not filled by the end of the
wait → sell at the bid then (taker fee).
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from propedge.models.kalshi_math import fee_per_contract

EPS = 1e-9


@dataclass(frozen=True)
class Quote:
    t0: datetime                 # first seen
    t1: datetime                 # last confirmed unchanged
    yes_bid: float | None
    yes_ask: float | None
    yes_bid_size: float | None = None
    yes_ask_size: float | None = None


class QuoteSeries:
    """One contract's stored quotes, sorted by first-seen time."""

    def __init__(self, quotes: list[Quote]):
        self.quotes = sorted(quotes, key=lambda q: q.t0)
        self._t0 = [q.t0 for q in self.quotes]

    def at(self, t: datetime, max_age_s: float) -> Quote | None:
        i = bisect.bisect_right(self._t0, t) - 1
        if i < 0:
            return None
        q = self.quotes[i]
        return q if (t - q.t1).total_seconds() <= max_age_s else None

    def during(self, start: datetime, end: datetime) -> list[Quote]:
        """Quotes first seen in (start, end]."""
        lo, hi = bisect.bisect_right(self._t0, start), bisect.bisect_right(self._t0, end)
        return self.quotes[lo:hi]


def side_prices(q: Quote, side: str) -> tuple[float | None, float | None, float | None]:
    """(bid, ask, contracts offered at the ask) for our side."""
    if side == "yes":
        return q.yes_bid, q.yes_ask, q.yes_ask_size
    bid = None if q.yes_ask is None else round(1 - q.yes_ask, 10)
    ask = None if q.yes_bid is None else round(1 - q.yes_bid, 10)
    return bid, ask, q.yes_bid_size


def yes_mid(q: Quote | None) -> float | None:
    if q is None or q.yes_bid is None or q.yes_ask is None:
        return None
    return (q.yes_bid + q.yes_ask) / 2


@dataclass(frozen=True)
class Fees:
    taker_coef: float
    maker_coef: float
    round_to: float
    contracts: float
    multiplier: float = 1.0
    maker_charged: bool = False          # only series with fee_type "quadratic_with_maker_fees"

    @classmethod
    def from_config(cls, cfg: dict, stat: str = "pts", fee_type: str | None = None) -> "Fees":
        k = cfg["odds"]["kalshi"]
        f = k["fees"]
        ft = fee_type or f.get("fee_type", {}).get(stat, "quadratic")
        return cls(f["taker_coef"], f["maker_coef"], f["round_to"], k["order_contracts"],
                   maker_charged=(ft == "quadratic_with_maker_fees"))

    def taker(self, price: float) -> float:
        if not 0 < price < 1:
            return 0.0
        return fee_per_contract(price, self.contracts, self.taker_coef, self.multiplier, self.round_to)

    def maker(self, price: float) -> float:
        if not self.maker_charged or not 0 < price < 1:
            return 0.0
        return fee_per_contract(price, self.contracts, self.maker_coef, self.multiplier, self.round_to)


def closest_to_half(rungs: list[tuple[str, float, Quote | None]], max_spread: float | None = None) -> str | None:
    """Ticker of the rung whose YES mid is closest to 0.50 (two-sided quotes only; tie → lower line).
    `max_spread`: only rungs whose YES spread is at most this — an empty book quoted 1¢/99¢ also has a
    50¢ mid, and it's no market at all."""
    best = None
    for ticker, line, q in rungs:
        mid = yes_mid(q)
        if mid is None or (max_spread is not None and q.yes_ask - q.yes_bid > max_spread + EPS):
            continue
        key = (round(abs(mid - 0.5), 9), line)
        if best is None or key < best[0]:
            best = (key, ticker)
    return None if best is None else best[1]


@dataclass(frozen=True)
class Trade:
    entry: float           # ask paid
    fee_in: float
    exit: float            # price received (settled value if decided)
    fee_out: float
    how: str               # bid | limit | limit_unfilled_bid | decided | no_bid
    ask_size: float | None

    @property
    def gross(self) -> float:
        return self.exit - self.entry

    @property
    def fees(self) -> float:
        return self.fee_in + self.fee_out

    @property
    def pnl(self) -> float:
        return self.exit - self.fee_out - self.entry - self.fee_in


Decided = Callable[[datetime], float | None]     # settled value of our side at t, or None if still open


def _sell_at_bid(series: QuoteSeries, side: str, t: datetime, fees: Fees, max_age_s: float,
                 decided: Decided | None) -> tuple[float, float, str] | None:
    v = decided(t) if decided else None
    if v is not None:
        return v, 0.0, "decided"                    # Kalshi charges no fee at settlement
    q = series.at(t, max_age_s)
    if q is None:
        return None
    bid, _, _ = side_prices(q, side)
    if bid is None or bid <= 0:
        return 0.0, 0.0, "no_bid"                   # nobody to sell to: worth nothing to us now
    return bid, fees.taker(bid), "bid"


def round_trip(series: QuoteSeries, side: str, t_entry: datetime, t_exit: datetime, fees: Fees,
               max_age_s: float, exit_mode: str = "bid", limit_wait_s: float = 60,
               decided: Decided | None = None) -> Trade | str:
    """Buy our side at its ask at t_entry; sell at t_exit. Returns a Trade, or the reason there's none."""
    q_in = series.at(t_entry, max_age_s)
    if q_in is None:
        return "no_entry_quote"
    if decided and decided(t_entry) is not None:
        return "decided_before_entry"
    _, ask, size = side_prices(q_in, side)
    if ask is None or not 0 < ask < 1:
        return "no_ask"
    fee_in = fees.taker(ask)

    if exit_mode == "limit" and not (decided and decided(t_exit) is not None):
        q = series.at(t_exit, max_age_s)
        if q is not None:
            bid, ask_x, _ = side_prices(q, side)
            if bid is not None and ask_x is not None and ask_x > bid + EPS:
                price = ask_x if ask_x - bid <= 0.01 + EPS else round(ask_x - 0.01, 10)
                end = t_exit + timedelta(seconds=limit_wait_s)
                for later in series.during(t_exit, end):
                    b, _, _ = side_prices(later, side)
                    if b is not None and b >= price - EPS:
                        return Trade(ask, fee_in, price, fees.maker(price), "limit", size)
                out = _sell_at_bid(series, side, end, fees, max_age_s, decided)
                if out is None:
                    return "no_exit_quote"
                return Trade(ask, fee_in, out[0], out[1], "limit_unfilled_bid" if out[2] == "bid" else out[2], size)
    out = _sell_at_bid(series, side, t_exit, fees, max_age_s, decided)
    if out is None:
        return "no_exit_quote"
    return Trade(ask, fee_in, out[0], out[1], out[2], size)


def bootstrap_ci(values, n: int = 2000, level: float = 0.90, seed: int = 7) -> tuple[float, float]:
    """Percentile interval for the mean, resampling the values (one per independent event)."""
    v = np.asarray([x for x in values if x == x], float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
