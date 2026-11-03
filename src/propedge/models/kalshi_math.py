"""Kalshi contract math: fees, all-in cost, edge and EV. Pure functions, no I/O.

A contract pays $1 if it wins and $0 if it loses. Buying at price P (dollars) means:
    profit if it wins = 1 − P − fee,  loss if it loses = P + fee
so with model win-probability p and all-in cost c = P + fee-per-contract:
    EV per contract  = p·(1 − c) − (1 − p)·c = p − c      ← this is our "edge"
    EV per $1 risked = (p − c) / c
There's no de-vig step: on an exchange we pay the ask, so the ask (plus fee) IS our price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def fee_for_order(price: float, contracts: float, coef: float, multiplier: float = 1.0,
                  round_to: float = 0.01) -> float:
    """Kalshi trading fee for one order: round_up(coef × multiplier × C × P × (1 − P)) to the cent.

    >>> fee_for_order(0.55, 100, 0.07)   # Kalshi's own example
    1.74
    """
    if not 0 < price < 1 or contracts <= 0:
        raise ValueError(f"bad order: price={price}, contracts={contracts}")
    raw = coef * multiplier * contracts * price * (1 - price)
    units = math.ceil(round(raw / round_to, 9))   # round() first so 1.75 doesn't become 1.76 via 1.7500000001
    return round(units * round_to, 10)


def fee_per_contract(price: float, contracts: float, coef: float, multiplier: float = 1.0,
                     round_to: float = 0.01) -> float:
    return fee_for_order(price, contracts, coef, multiplier, round_to) / contracts


@dataclass(frozen=True)
class SidePrice:
    side: str            # "yes" | "no"
    model_prob: float    # our probability that this side wins
    ask: float           # what we'd pay per contract
    fee: float           # fee per contract at our order size
    cost: float          # ask + fee
    edge: float          # model_prob − cost  (= EV per contract, in dollars)
    ev_per_dollar: float # edge / cost


def price_side(side: str, model_prob: float, ask: float | None, contracts: float, coef: float,
               multiplier: float = 1.0, round_to: float = 0.01) -> SidePrice | None:
    """Edge and EV of taking one side at its ask. None if nobody is offering that side."""
    if ask is None or not 0 < ask < 1:
        return None
    if not 0 <= model_prob <= 1:
        raise ValueError(f"model_prob out of range: {model_prob}")
    fee = fee_per_contract(ask, contracts, coef, multiplier, round_to)
    cost = ask + fee
    edge = model_prob - cost
    return SidePrice(side, model_prob, ask, fee, cost, edge, edge / cost)


def price_both_sides(p_yes: float, yes_bid: float | None, yes_ask: float | None, contracts: float,
                     coef: float, multiplier: float = 1.0, round_to: float = 0.01) -> list[SidePrice]:
    """YES costs the yes ask; NO costs 1 − yes bid (a NO bid at x is a YES ask at 1 − x)."""
    no_ask = None if yes_bid is None else round(1 - yes_bid, 10)
    sides = [price_side("yes", p_yes, yes_ask, contracts, coef, multiplier, round_to),
             price_side("no", 1 - p_yes, no_ask, contracts, coef, multiplier, round_to)]
    return [s for s in sides if s is not None]
