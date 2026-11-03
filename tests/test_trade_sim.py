"""Buy-then-sell simulation on recorded quotes, and event detection from stored snapshots."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from propedge.edge.event_trades import BoxLine, Frame, detect_events
from propedge.edge.scoreboard import in_nights
from propedge.models.trade_sim import Fees, Quote, QuoteSeries, bootstrap_ci, closest_to_half, round_trip

T0 = datetime(2026, 10, 24, 23, 0, tzinfo=timezone.utc)
FEES = Fees(taker_coef=0.07, maker_coef=0.0175, round_to=0.01, contracts=10)


def s(sec):
    return T0 + timedelta(seconds=sec)


def q(a, b, bid, ask, bs=20, as_=20):
    """Quote first seen at a s, confirmed until b s."""
    return Quote(s(a), s(b), bid, ask, bs, as_)


def test_fees_taker_rounding_and_no_maker_fee_for_quadratic():
    assert FEES.taker(0.50) == pytest.approx(0.018)       # 0.07·10·.25 = 0.175 → 0.18 per 10
    assert FEES.maker(0.50) == 0.0                         # fee_type "quadratic": no maker fee
    assert Fees(0.07, 0.0175, 0.01, 10, maker_charged=True).maker(0.50) == pytest.approx(0.005)
    assert FEES.taker(0.0) == 0.0 and FEES.taker(1.0) == 0.0


def test_quote_must_be_fresh():
    ser = QuoteSeries([q(0, 10, 0.40, 0.44)])
    assert ser.at(s(5), 30) is not None
    assert ser.at(s(39), 30) is not None
    assert ser.at(s(41), 30) is None        # last confirmed 31 s earlier: a polling gap, not a price
    assert ser.at(s(-1), 30) is None


def test_round_trip_no_side_buys_at_one_minus_bid_sells_at_one_minus_ask():
    ser = QuoteSeries([q(0, 59, 0.40, 0.44), q(60, 200, 0.30, 0.33)])
    tr = round_trip(ser, "no", s(10), s(70), FEES, 30)
    assert tr.entry == pytest.approx(0.60) and tr.exit == pytest.approx(0.67)
    assert tr.pnl == pytest.approx(0.67 - FEES.taker(0.67) - 0.60 - FEES.taker(0.60))
    assert tr.how == "bid"


def test_round_trip_missing_quotes_and_empty_bid():
    ser = QuoteSeries([q(0, 10, 0.40, 0.44)])
    assert round_trip(ser, "no", s(100), s(160), FEES, 30) == "no_entry_quote"
    assert round_trip(ser, "no", s(5), s(100), FEES, 30) == "no_exit_quote"
    ser = QuoteSeries([q(0, 59, 0.40, 0.44), q(60, 200, 0.95, None)])      # nobody sells YES = no NO bid
    tr = round_trip(ser, "no", s(5), s(70), FEES, 30)
    assert tr.exit == 0.0 and tr.how == "no_bid"


def test_decided_rung_settles_without_fee():
    ser = QuoteSeries([q(0, 300, 0.40, 0.44)])
    tr = round_trip(ser, "no", s(5), s(65), FEES, 30, decided=lambda t: 0.0 if t >= s(50) else None)
    assert tr.exit == 0.0 and tr.fee_out == 0.0 and tr.how == "decided"
    assert round_trip(ser, "no", s(60), s(90), FEES, 30, decided=lambda t: 0.0) == "decided_before_entry"


def test_limit_sell_fills_only_when_a_recorded_bid_reaches_it():
    # at exit (60 s): YES 0.30/0.34 -> NO bid 0.66, NO ask 0.70 -> our resting NO sell at 0.69
    base = [q(0, 59, 0.40, 0.44), q(60, 69, 0.30, 0.34)]
    filled = QuoteSeries(base + [q(70, 200, 0.29, 0.31)])         # NO bid 0.69 ≥ 0.69 → filled
    tr = round_trip(filled, "no", s(5), s(60), FEES, 30, "limit", 60)
    assert tr.how == "limit" and tr.exit == pytest.approx(0.69) and tr.fee_out == 0.0
    unfilled = QuoteSeries(base + [q(70, 200, 0.31, 0.32)])       # NO bid 0.68 < 0.69 → fall back to bid
    tr = round_trip(unfilled, "no", s(5), s(60), FEES, 30, "limit", 60)
    assert tr.how == "limit_unfilled_bid" and tr.exit == pytest.approx(0.68) and tr.fee_out > 0


def test_limit_sell_at_the_ask_when_spread_is_one_cent():
    ser = QuoteSeries([q(0, 59, 0.40, 0.44), q(60, 69, 0.30, 0.31), q(70, 200, 0.29, 0.30)])
    tr = round_trip(ser, "no", s(5), s(60), FEES, 30, "limit", 60)
    assert tr.how == "limit" and tr.exit == pytest.approx(0.70)


def test_closest_to_half_ignores_empty_books_with_a_spread_limit():
    rungs = [("A", 9.5, Quote(T0, T0, 0.70, 0.74)), ("B", 14.5, Quote(T0, T0, 0.01, 0.99)),
             ("C", 19.5, Quote(T0, T0, 0.40, 0.44)), ("D", 24.5, Quote(T0, T0, None, 0.10))]
    assert closest_to_half(rungs) == "B"                    # 1¢/99¢ has a 50¢ mid…
    assert closest_to_half(rungs, max_spread=0.06) == "C"   # …but it's no market
    tie = [("X", 19.5, Quote(T0, T0, 0.48, 0.52)), ("Y", 14.5, Quote(T0, T0, 0.48, 0.52))]
    assert closest_to_half(tie) == "Y"                      # tie → lower line


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([0.01, 0.02, 0.03, 0.04, 0.05])
    assert lo < 0.03 < hi
    assert all(x != x for x in bootstrap_ci([0.1]))


def box(pf1=0, st=True):
    return {1: BoxLine("HHH", st, pf1, 10), 2: BoxLine("HHH", False, 0, 4), 3: BoxLine("AAA", True, 1, 8)}


def fr(sec, period, home, away, b=None, status="in", clock=300.0):
    return Frame(s(sec), status, period, clock, home, away, b)


def test_detect_foul4_and_blowout_from_snapshots():
    frames = [fr(0, 1, 2, 0, box(3)), fr(20, 2, 30, 20), fr(40, 2, 32, 20, box(4)),     # 3 -> 4 fouls
              fr(60, 3, 60, 41), fr(80, 3, 62, 41), fr(100, 3, 62, 30), fr(120, 3, 70, 49)]
    ev = detect_events("2026-10-24_AAA_HHH", frames)
    assert [(e.type, e.t) for e in ev] == [("foul4", s(40)), ("blowout", s(80))]
    assert ev[0].players == {1: "fouler"} and ev[0].period == 2
    assert ev[1].players == {1: "starter", 3: "starter"} and ev[1].lead == 21     # starters, both teams


def test_first_frame_and_late_start_are_not_events():
    # polling starts with 4 fouls already and a 25-point second-half lead: neither is an event, and the
    # blowout doesn't fire later when the lead dips and comes back
    frames = [fr(0, 3, 70, 45, box(4)), fr(20, 3, 70, 52), fr(40, 3, 75, 52), fr(60, 3, 75, 53, box(5))]
    assert detect_events("2026-10-24_AAA_HHH", frames) == []


def test_first_half_lead_is_not_a_blowout():
    frames = [fr(0, 2, 30, 15, box()), fr(20, 2, 40, 15, clock=200.0)]
    assert detect_events("2026-10-24_AAA_HHH", frames) == []


def test_nights_filter_by_game_date_not_utc():
    ids = pd.Series(["2026-10-23_LAL_GSW", "2026-10-24_WAS_CHI", "2026-10-25_ORL_NYK"])
    assert in_nights(ids, pd.Timestamp("2026-10-24").date(), pd.Timestamp("2026-10-24").date()).tolist() == \
        [False, True, False]


def test_rescinded_fourth_foul_counts_once():
    frames = [fr(0, 2, 30, 28, box(3)), fr(20, 3, 32, 28, box(4)), fr(40, 3, 32, 30, box(3)),
              fr(60, 4, 40, 38, box(4))]
    ev = detect_events("2026-10-25_AAA_HHH", frames)
    assert [(e.type, e.t, e.period) for e in ev] == [("foul4", s(20), 3)]
