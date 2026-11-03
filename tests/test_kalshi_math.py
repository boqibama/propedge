"""Fee, edge and EV math. These numbers decide real money, so they're tested exhaustively."""

import pytest

from propedge.models.kalshi_math import fee_for_order, fee_per_contract, price_both_sides, price_side

TAKER = 0.07


@pytest.mark.parametrize("price,contracts,expected", [
    (0.55, 100, 1.74),   # Kalshi's published example
    (0.50, 1, 0.02),     # 0.0175 rounds UP to 2 cents
    (0.50, 10, 0.18),    # 0.175 -> 0.18
    (0.50, 100, 1.75),   # exactly 1.75: must not round up to 1.76 via float error
    (0.99, 10, 0.01),    # 0.00693 -> 1 cent minimum
    (0.01, 1, 0.01),
    (0.20, 10, 0.12),    # 0.112 -> 0.12
])
def test_fee_for_order(price, contracts, expected):
    assert fee_for_order(price, contracts, TAKER) == pytest.approx(expected)


def test_fee_is_symmetric_and_peaks_at_50_cents():
    assert fee_for_order(0.30, 50, TAKER) == fee_for_order(0.70, 50, TAKER)
    fees = [fee_for_order(p / 100, 1000, TAKER) for p in range(1, 100)]
    assert max(fees) == fee_for_order(0.50, 1000, TAKER)


def test_fee_multiplier_and_maker_coefficient():
    assert fee_for_order(0.50, 100, TAKER, multiplier=2) == pytest.approx(3.50)
    assert fee_for_order(0.50, 100, 0.0175) == pytest.approx(0.44)   # 0.4375 -> 0.44


def test_fee_rejects_bad_orders():
    for price in (0, 1, -0.1, 1.2):
        with pytest.raises(ValueError):
            fee_for_order(price, 10, TAKER)
    with pytest.raises(ValueError):
        fee_for_order(0.5, 0, TAKER)


def test_fee_per_contract_shrinks_with_order_size():
    assert fee_per_contract(0.50, 1, TAKER) == pytest.approx(0.02)
    assert fee_per_contract(0.50, 100, TAKER) == pytest.approx(0.0175)


def test_edge_and_ev_definitions():
    s = price_side("yes", 0.60, 0.50, 10, TAKER)
    assert s.fee == pytest.approx(0.018)
    assert s.cost == pytest.approx(0.518)
    assert s.edge == pytest.approx(0.60 - 0.518)
    assert s.ev_per_dollar == pytest.approx((0.60 - 0.518) / 0.518)
    # EV per contract computed the long way: win (1 - cost) with p, lose cost with 1 - p
    assert s.edge == pytest.approx(0.60 * (1 - s.cost) - 0.40 * s.cost)


def test_fair_price_has_negative_edge_after_fees():
    s = price_side("yes", 0.50, 0.50, 10, TAKER)
    assert s.edge < 0


def test_no_side_uses_one_minus_yes_bid():
    sides = {s.side: s for s in price_both_sides(0.30, yes_bid=0.35, yes_ask=0.40, contracts=10, coef=TAKER)}
    assert sides["no"].ask == pytest.approx(0.65)
    assert sides["no"].model_prob == pytest.approx(0.70)
    assert sides["no"].edge == pytest.approx(0.70 - 0.65 - fee_per_contract(0.65, 10, TAKER))
    assert sides["yes"].edge < 0


def test_missing_book_side_is_skipped():
    assert [s.side for s in price_both_sides(0.5, None, 0.40, 10, TAKER)] == ["yes"]
    assert [s.side for s in price_both_sides(0.5, 0.40, None, 10, TAKER)] == ["no"]
    assert price_side("yes", 0.5, None, 10, TAKER) is None


def test_model_prob_must_be_a_probability():
    with pytest.raises(ValueError):
        price_side("yes", 1.2, 0.5, 10, TAKER)
