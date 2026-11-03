from datetime import datetime, timedelta, timezone

from propedge.config import load_config
from propedge.models.suppression import (QuotePoint, detect_moves, game_move_reason, market_reasons,
                                         player_move_reason, scoring_window_reason)

NOW = datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)
CFG = load_config()


def ago(s):
    return NOW - timedelta(seconds=s)


def steady(price=0.50):
    return [QuotePoint(ago(600), price - 0.01, price + 0.01)]


def jumped(by=0.10, seconds_ago=30):
    return [QuotePoint(ago(600), 0.49, 0.51), QuotePoint(ago(seconds_ago), 0.49 + by, 0.51 + by)]


def test_detects_recent_sudden_move():
    moves = detect_moves(jumped(0.10, 30), NOW, window_s=90, lookback_s=120, threshold=0.05, max_spread=0.1)
    assert len(moves) == 1 and abs(moves[0][1] - 0.10) < 1e-9


def test_old_move_or_small_move_is_ignored():
    assert not detect_moves(jumped(0.10, 300), NOW, 90, 120, 0.05, 0.1)
    assert not detect_moves(jumped(0.03, 30), NOW, 90, 120, 0.05, 0.1)


def test_wide_rung_moves_are_ignored():
    pts = [QuotePoint(ago(600), 0.10, 0.40), QuotePoint(ago(30), 0.30, 0.60)]
    assert not detect_moves(pts, NOW, 90, 120, 0.05, 0.10)


def test_player_move_reason_names_the_contract():
    r = player_move_reason({"T-25": steady(), "T-30": jumped(-0.08)}, NOW, CFG)
    assert r.code == "price_move_player" and "T-30" in r.detail and "-0.08" in r.detail
    assert player_move_reason({"T-25": steady()}, NOW, CFG) is None


def test_game_move_needs_several_players():
    min_players = CFG["edge"]["no_flag"]["game_move_min_players"]
    few = {pid: {"t": jumped()} for pid in range(min_players - 1)}
    few[99] = {"t": steady()}
    assert game_move_reason(few, NOW, CFG) is None
    many = {pid: {"t": jumped()} for pid in range(min_players)}
    r = game_move_reason(many, NOW, CFG)
    assert r.code == "price_move_game"


def test_scoring_window():
    window = CFG["edge"]["no_flag"]["after_scoring_play_s"]
    assert scoring_window_reason([ago(window - 5)], NOW, CFG).code == "scoring_window"
    assert scoring_window_reason([ago(window + 5)], NOW, CFG) is None


def test_market_reasons():
    liq = CFG["odds"]["liquidity"]
    ok = market_reasons(0.48, 0.50, 100, 100, "yes", ago(5), NOW, CFG)
    assert ok == []
    wide = market_reasons(0.40, 0.40 + liq["max_spread"] + 0.02, 100, 100, "yes", ago(5), NOW, CFG)
    assert [r.code for r in wide] == ["thin_market"]
    thin_no = market_reasons(0.48, 0.50, 1, 100, "no", ago(5), NOW, CFG)   # NO size = YES bid size
    assert [r.code for r in thin_no] == ["thin_market"]
    stale = market_reasons(0.48, 0.50, 100, 100, "yes", ago(CFG["odds"]["stale_after_s"] + 10), NOW, CFG)
    assert [r.code for r in stale] == ["stale_quote"]
