"""Identity helpers: one canonical key for each team, player and game across NBA, ESPN and Kalshi.

Canonical choices:
- Team  = NBA tricode ("GSW", "NYK", ...). ESPN uses a few different codes; Kalshi uses tricodes.
- Game  = "<ET date>_<away>_<home>", e.g. "2026-10-21_GSW_LAL". The date is the US-Eastern
          calendar date, because that's how the NBA and Kalshi label games (a 7pm PT tip is
          already the next day in UTC).
- Player names are compared via `name_key` (accents, punctuation and suffixes removed).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

# ESPN abbreviation -> NBA tricode, for the teams where they differ.
ESPN_TO_TRICODE = {
    "GS": "GSW",
    "NY": "NYK",
    "SA": "SAS",
    "NO": "NOP",
    "UTAH": "UTA",
    "WSH": "WAS",
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def tricode_from_espn(abbr: str) -> str:
    return ESPN_TO_TRICODE.get(abbr.upper(), abbr.upper())


def name_key(name: str) -> str:
    """Normalise a player name for matching: 'Luka Dončić' -> 'luka doncic', 'Jaren Jackson Jr.' -> 'jaren jackson'."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = re.sub(r"[.'’`-]", "", text.lower())
    words = [w for w in re.split(r"\s+", text) if w and w not in _SUFFIXES]
    return " ".join(words)


def eastern_date(ts: datetime) -> date:
    """The US-Eastern calendar date of a timezone-aware timestamp."""
    return ts.astimezone(EASTERN).date()


def game_key(game_date: date, away: str, home: str) -> str:
    return f"{game_date.isoformat()}_{away}_{home}"


_KALSHI_EVENT = re.compile(r"^(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<away>[A-Z]{3})(?P<home>[A-Z]{3})$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def parse_kalshi_event(event_ticker: str) -> tuple[date, str, str] | None:
    """'KXNBAPTS-26OCT21GSWLAL' -> (2026-10-21, 'GSW', 'LAL'). None if the format is unexpected."""
    suffix = event_ticker.split("-", 1)[-1]
    m = _KALSHI_EVENT.match(suffix)
    if not m or m["mon"] not in _MONTHS:
        return None
    return date(2000 + int(m["yy"]), _MONTHS[m["mon"]], int(m["dd"])), m["away"], m["home"]


def season_for(game_date: date) -> str:
    """NBA season label for a date: games from October onward belong to 'YYYY-(YY+1)'."""
    start = game_date.year if game_date.month >= 8 else game_date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def clock_to_seconds(display: str | None) -> float | None:
    """Game clock text -> seconds left in the period. '7:42' -> 462, '2.0' -> 2.0, '12:00' -> 720."""
    if not display:
        return None
    try:
        if ":" in display:
            minutes, seconds = display.split(":")
            return int(minutes) * 60 + float(seconds)
        return float(display)
    except ValueError:
        return None


def minutes_to_float(value: str | int | float | None) -> float | None:
    """Box-score minutes in any of the formats we see ('34', '34:12', 34.2) -> 34.2."""
    if value in (None, "", "--"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if ":" in value:
        m, s = value.split(":")
        return int(m) + int(s) / 60
    try:
        return float(value)
    except ValueError:
        return None
