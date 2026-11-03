"""Replay recorded live payloads (raw ESPN summaries + Kalshi market lists) through the normal
providers' interfaces, on a virtual clock. Lets us develop and test when no games are on.

Everything replayed is marked `is_mock=True` and the provider is named "replay", and the CLI
writes replays to a separate database, so replayed data can never be mistaken for live data.

Recording layout (one directory per night):
    espn_<event id>_<YYYYmmddTHHMMSSZ>.json.gz      {"fetched_at": iso, "data": <summary payload>}
    kalshi_<series>_<YYYYmmddTHHMMSSZ>.json.gz      {"fetched_at": iso, "data": <markets payload>}
"""

from __future__ import annotations

import bisect
import gzip
import json
import re
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from propedge.data.espn import parse_summary
from propedge.data.live_base import LiveGameState, LiveStatsProvider, ScheduledGame
from propedge.odds.base import OddsProvider, PropQuote
from propedge.odds.kalshi import parse_market

_FILE = re.compile(r"^(?P<kind>espn|kalshi)_(?P<key>[A-Za-z0-9]+)_(?P<ts>\d{8}T\d{6}Z)\.json\.gz$")


def _load(path: Path) -> tuple[datetime, dict]:
    with gzip.open(path, "rt") as f:
        blob = json.load(f)
    return datetime.fromisoformat(blob["fetched_at"]), blob["data"]


class ReplayClock:
    """Virtual time: `sleep` advances it instantly, so a 2-hour recording replays in seconds."""

    def __init__(self, start: datetime):
        self.t = start

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


class _Recording:
    def __init__(self, directory: Path):
        self.files: dict[tuple[str, str], list[tuple[datetime, Path]]] = {}
        for p in sorted(directory.iterdir()):
            m = _FILE.match(p.name)
            if m:
                ts = datetime.strptime(m["ts"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                self.files.setdefault((m["kind"], m["key"]), []).append((ts, p))
        if not self.files:
            raise FileNotFoundError(f"no recording files in {directory}")

    @property
    def start(self) -> datetime:
        return min(v[0][0] for v in self.files.values())

    @property
    def end(self) -> datetime:
        return max(v[-1][0] for v in self.files.values())

    def latest(self, kind: str, key: str, now: datetime) -> Path | None:
        items = self.files.get((kind, key), [])
        i = bisect.bisect_right([t for t, _ in items], now)
        return items[i - 1][1] if i else None

    def keys(self, kind: str) -> list[str]:
        return [k for (kd, k) in self.files if kd == kind]


class ReplayLiveProvider(LiveStatsProvider):
    name = "replay"
    is_mock = True

    def __init__(self, directory: Path, clock: ReplayClock):
        self.rec = _Recording(directory)
        self.clock = clock

    def _state(self, gid: str) -> LiveGameState | None:
        path = self.rec.latest("espn", gid, self.clock.now())
        if path is None:
            return None
        fetched_at, data = _load(path)
        return replace(parse_summary(data, fetched_at, provider="replay"), is_mock=True)

    def scoreboard(self, day: date | None = None) -> list[ScheduledGame]:
        games = []
        for gid in self.rec.keys("espn"):
            state = self._state(gid)
            if state is None:
                continue
            last_file = self.rec.files[("espn", gid)][-1][0]
            # Recordings only cover live play; once a game's files run out, treat it as over.
            status = "post" if self.clock.now() > last_file + timedelta(minutes=2) else state.status
            games.append(ScheduledGame(gid, state.game_date, None, state.away, state.home, status,
                                       state.away_score, state.home_score))
        return games

    def game_state(self, provider_game_id: str) -> LiveGameState | None:
        return self._state(provider_game_id)


class ReplayOddsProvider(OddsProvider):
    name = "replay"
    is_mock = True

    def __init__(self, directory: Path, clock: ReplayClock, series: dict[str, str]):
        self.rec = _Recording(directory)
        self.clock = clock
        self.series = series

    def fetch_quotes(self, stats: list[str]) -> list[PropQuote]:
        quotes = []
        for stat in stats:
            path = self.rec.latest("kalshi", self.series[stat], self.clock.now())
            if path is None:
                continue
            fetched_at, data = _load(path)
            for m in data.get("markets", []):
                q = parse_market(m, stat, fetched_at, provider="replay")
                if q:
                    quotes.append(replace(q, is_mock=True))
        return quotes
