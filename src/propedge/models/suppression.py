"""No-flag rules: reasons a would-be flag must NOT be acted on. Pure functions, no I/O.

Each rule returns `Reason`s; a would-be flag with any reason is logged as "suppressed".
The windows come from config.yaml (edge.no_flag, odds.liquidity) and were sized from the
measured ~45–50 s feed delay: the market often knows about a play before we do.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Reason:
    code: str      # low_confidence | thin_market | stale_quote | price_move_player | price_move_game |
                   # scoring_window | feed_lagging | injury_status
    detail: str

    def as_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class QuotePoint:
    """One stored quote for a contract (odds_snapshots row): valid from `t` until the next point."""
    t: datetime
    yes_bid: float | None
    yes_ask: float | None


def mid(bid: float | None, ask: float | None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return bid if bid is not None else ask


def _mid_at(points: list[QuotePoint], t: datetime) -> float | None:
    """Mid of the quote that was current at time t (points sorted by t)."""
    current = None
    for p in points:
        if p.t <= t:
            current = p
        else:
            break
    return None if current is None else mid(current.yes_bid, current.yes_ask)


def detect_moves(points: list[QuotePoint], now: datetime, window_s: float, lookback_s: float,
                 threshold: float, max_spread: float) -> list[tuple[datetime, float]]:
    """Sudden moves in one contract: for every quote change in the last `window_s`, compare its mid
    with the mid `lookback_s` earlier. Returns [(time, signed move)] where |move| ≥ threshold.
    Quotes wider than `max_spread` are ignored (a thin rung's mid jumps without any news)."""
    pts = sorted(points, key=lambda p: p.t)
    moves = []
    for p in pts:
        if not now - timedelta(seconds=window_s) <= p.t <= now:
            continue
        if p.yes_bid is None or p.yes_ask is None or p.yes_ask - p.yes_bid > max_spread:
            continue
        before = _mid_at(pts, p.t - timedelta(seconds=lookback_s))
        if before is None:
            continue
        move = mid(p.yes_bid, p.yes_ask) - before
        if abs(move) >= threshold - 1e-12:
            moves.append((p.t, move))
    return moves


def player_move_reason(histories: dict[str, list[QuotePoint]], now: datetime, cfg: dict) -> Reason | None:
    """Any of the player's contracts moved suddenly within `after_price_move_s`."""
    nf = cfg["edge"]["no_flag"]
    worst = None
    for ticker, pts in histories.items():
        for t, move in detect_moves(pts, now, nf["after_price_move_s"], nf["price_move_lookback_s"],
                                    nf["price_move_threshold"], nf["move_max_spread"]):
            if worst is None or abs(move) > abs(worst[2]):
                worst = (ticker, t, move)
    if worst is None:
        return None
    ticker, t, move = worst
    return Reason("price_move_player", f"{ticker} moved {move:+.2f} at {t:%H:%M:%S}")


def game_move_reason(histories_by_player: dict[int, dict[str, list[QuotePoint]]], now: datetime,
                     cfg: dict) -> Reason | None:
    """Several players in one game moved within `after_game_move_s` -> probably news we haven't seen."""
    nf = cfg["edge"]["no_flag"]
    movers = []
    for player_id, histories in histories_by_player.items():
        if any(detect_moves(pts, now, nf["after_game_move_s"], nf["price_move_lookback_s"],
                            nf["game_move_threshold"], nf["move_max_spread"]) for pts in histories.values()):
            movers.append(player_id)
    if len(movers) >= nf["game_move_min_players"]:
        return Reason("price_move_game", f"{len(movers)} players' prices moved within "
                                         f"{nf['after_game_move_s']}s (players {sorted(movers)})")
    return None


def scoring_window_reason(points_changed_at: list[datetime], now: datetime, cfg: dict) -> Reason | None:
    """The player's box-score points changed within `after_scoring_play_s` (live only)."""
    window = timedelta(seconds=cfg["edge"]["no_flag"]["after_scoring_play_s"])
    recent = [t for t in points_changed_at if now - window <= t <= now]
    if recent:
        return Reason("scoring_window", f"points changed {int((now - max(recent)).total_seconds())}s ago")
    return None


def market_reasons(yes_bid: float | None, yes_ask: float | None, yes_bid_size: float | None,
                   yes_ask_size: float | None, side: str, last_confirmed: datetime, now: datetime,
                   cfg: dict) -> list[Reason]:
    """Liquidity (spread, size at the ask we'd take) and quote freshness."""
    liq, reasons = cfg["odds"]["liquidity"], []
    if yes_bid is None or yes_ask is None:
        reasons.append(Reason("thin_market", "one side of the book is empty"))
    elif yes_ask - yes_bid > liq["max_spread"] + 1e-9:
        reasons.append(Reason("thin_market", f"spread {yes_ask - yes_bid:.2f} > {liq['max_spread']}"))
    # Size at the ask we'd take: YES ask size for YES; for NO it's the YES bid size
    # (a resting YES bid at x is the same order as a NO offer at 1 − x).
    size = yes_ask_size if side == "yes" else yes_bid_size
    if size is not None and size < liq["min_ask_size"]:
        reasons.append(Reason("thin_market", f"only {size:.0f} contracts at the {side} ask"))
    age = (now - last_confirmed).total_seconds()
    if age > cfg["odds"]["stale_after_s"]:
        reasons.append(Reason("stale_quote", f"quote last confirmed {age:.0f}s ago"))
    return reasons
