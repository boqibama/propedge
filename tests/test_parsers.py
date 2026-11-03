"""ESPN, Kalshi and NBA game-log parsing against trimmed real payloads (tests/fixtures/)."""

import json
from datetime import date, datetime, timezone
from pathlib import Path

from propedge.data.espn import parse_scoreboard, parse_summary
from propedge.ingest.history import _home_away, game_sides
from propedge.odds.kalshi import parse_market

FIX = Path(__file__).parent / "fixtures"
FETCHED = datetime(2026, 10, 22, 4, 31, 30, tzinfo=timezone.utc)


def load(name):
    return json.loads((FIX / name).read_text())


# ----------------------------------------------------------------------------- ESPN

def test_summary_game_level_fields():
    s = parse_summary(load("espn_summary_live.json"), FETCHED)
    assert (s.away, s.home) == ("GSW", "LAL")          # ESPN "GS" mapped to the NBA tricode
    assert s.game_date == date(2026, 10, 21)           # 02:00 UTC tip = Oct 21 Eastern
    assert s.status == "in" and s.period == 4 and s.clock_s == 4 * 60 + 33
    assert s.home_spread == -6.5 and s.total == 233.5


def test_summary_players_have_minutes_fouls_and_shooting_splits():
    s = parse_summary(load("espn_summary_live.json"), FETCHED)
    played = [p for p in s.players if not p.did_not_play]
    assert len(played) >= 16
    for p in played:
        assert p.minutes is not None and p.pf is not None and p.pts is not None
        assert p.pts == 2 * (p.fgm - p.fg3m) + 3 * p.fg3m + p.ftm   # box score adds up
    dnp = [p for p in s.players if p.did_not_play]
    assert dnp and all(p.minutes is None for p in dnp)


def test_summary_plays_and_feed_lag():
    s = parse_summary(load("espn_summary_live.json"), FETCHED)
    assert len(s.plays) == 40
    assert [p.sequence for p in s.plays] == sorted(p.sequence for p in s.plays)
    scoring = [p for p in s.plays if p.scoring]
    assert scoring and all(p.score_value in (1, 2, 3) for p in scoring)
    assert all(p.team in ("GSW", "LAL", None) for p in s.plays)
    # newest play in the fixture is 04:30:32Z; fetched 04:31:30Z -> 58 s behind
    assert s.feed_lag_s == 58


def test_scoreboard():
    games = parse_scoreboard(load("espn_scoreboard.json"))
    assert games[0].away == "ATL" and games[0].home == "ORL" and games[0].status in ("pre", "in", "post")


# ----------------------------------------------------------------------------- Kalshi

def test_kalshi_ladder_rung_becomes_over_line():
    m = load("kalshi_markets.json")["markets"][0]          # "Stephen Curry: 35+ points"
    q = parse_market(m, "pts", FETCHED)
    assert q.player_name == "Stephen Curry" and q.team == "GSW"
    assert q.line == 34.5                                  # 35+ == over 34.5
    assert (q.game_date, q.away, q.home) == (date(2026, 10, 21), "GSW", "LAL")


def test_kalshi_prices_and_no_side():
    m = dict(load("kalshi_markets.json")["markets"][0], yes_bid_dollars="0.4200", yes_ask_dollars="0.4700")
    q = parse_market(m, "pts", FETCHED)
    assert q.yes_bid == 0.42 and q.yes_ask == 0.47
    assert q.no_ask == 0.58 and q.spread == 0.05


def test_kalshi_empty_book_sides_become_none():
    m = dict(load("kalshi_markets.json")["markets"][0], yes_bid_dollars="0.0000", yes_ask_dollars="1.0000")
    q = parse_market(m, "pts", FETCHED)
    assert q.yes_bid is None and q.yes_ask is None and q.no_ask is None and q.spread is None


def test_kalshi_unknown_strike_type_is_skipped():
    m = dict(load("kalshi_markets.json")["markets"][0], strike_type="between")
    assert parse_market(m, "pts", FETCHED) is None


# ----------------------------------------------------------------------------- NBA game logs

def _row(gid, team, matchup):
    return {"GAME_ID": gid, "TEAM_ABBREVIATION": team, "MATCHUP": matchup}


def test_game_sides_normal_game():
    rows = [_row("1", "LAL", "LAL vs. GSW"), _row("1", "GSW", "GSW @ LAL")]
    sides = game_sides(rows)
    assert sides["1"] == ("GSW", "LAL", False)
    assert _home_away(rows[0], sides) == (True, "GSW", "GSW", "LAL")
    assert _home_away(rows[1], sides) == (False, "LAL", "GSW", "LAL")


def test_game_sides_neutral_site_both_rows_say_at():
    # Real case: 2025-26 game 0022500147 lists 'DAL @ DET' and 'DET @ DAL'.
    rows = [_row("2", "DAL", "DAL @ DET"), _row("2", "DET", "DET @ DAL")]
    sides = game_sides(rows)
    assert sides["2"] == ("DAL", "DET", True)
    assert _home_away(rows[0], sides)[0] is False and _home_away(rows[1], sides)[0] is False
