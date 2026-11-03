from datetime import date, datetime, timezone

from propedge.ids import (clock_to_seconds, eastern_date, game_key, minutes_to_float, name_key,
                          parse_kalshi_event, season_for, tricode_from_espn)


def test_name_key_strips_accents_punctuation_and_suffixes():
    assert name_key("Luka Dončić") == "luka doncic"
    assert name_key("Jaren Jackson Jr.") == "jaren jackson"
    assert name_key("De'Anthony Melton") == "deanthony melton"
    assert name_key("Karl-Anthony Towns") == "karlanthony towns"
    assert name_key("Gary Trent Jr") == name_key("Gary Trent")
    assert name_key("  Alperen   Şengün ") == "alperen sengun"


def test_espn_abbreviations_map_to_nba_tricodes():
    assert tricode_from_espn("GS") == "GSW"
    assert tricode_from_espn("UTAH") == "UTA"
    assert tricode_from_espn("WSH") == "WAS"
    assert tricode_from_espn("LAL") == "LAL"


def test_late_west_coast_tip_belongs_to_previous_eastern_date():
    # 7pm PT on Oct 21 is 02:00 UTC on Oct 22, but the NBA and Kalshi call it an Oct 21 game.
    assert eastern_date(datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)) == date(2026, 10, 21)
    assert game_key(date(2026, 10, 21), "GSW", "LAL") == "2026-10-21_GSW_LAL"


def test_parse_kalshi_event():
    assert parse_kalshi_event("KXNBAPTS-26OCT21GSWLAL") == (date(2026, 10, 21), "GSW", "LAL")
    assert parse_kalshi_event("KXNBAPTS-weird") is None


def test_clock_and_minutes_parsing():
    assert clock_to_seconds("7:42") == 462
    assert clock_to_seconds("12:00") == 720
    assert clock_to_seconds("2.0") == 2.0
    assert clock_to_seconds(None) is None
    assert minutes_to_float("34") == 34.0
    assert minutes_to_float("34:30") == 34.5
    assert minutes_to_float("--") is None


def test_season_for():
    assert season_for(date(2026, 10, 21)) == "2026-27"
    assert season_for(date(2027, 4, 10)) == "2026-27"
