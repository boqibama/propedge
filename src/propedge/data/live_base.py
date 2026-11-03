"""LiveStatsProvider: the swappable interface for live game data.

ESPN is the current implementation (cdn.nba.com is blocked on our network). Everything
downstream only sees these dataclasses, never a provider's raw JSON, so switching providers
means writing one new class.

Player and team identities here are the PROVIDER's (e.g. ESPN athlete ids); the poller
maps them to our own `players.id`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class ScheduledGame:
    provider_game_id: str
    game_date: date              # US-Eastern date
    start_time: datetime | None
    away: str                    # NBA tricode
    home: str
    status: str                  # pre | in | post
    away_score: int | None = None
    home_score: int | None = None
    home_spread: float | None = None   # current line when the provider shows one (−5.5 = home favoured)
    total: float | None = None


@dataclass(frozen=True)
class LivePlayerLine:
    provider_player_id: str
    name: str
    team: str
    position: str | None
    starter: bool | None
    did_not_play: bool
    minutes: float | None
    pts: int | None = None
    fgm: int | None = None
    fga: int | None = None
    fg3m: int | None = None
    fg3a: int | None = None
    ftm: int | None = None
    fta: int | None = None
    reb: int | None = None
    ast: int | None = None
    stl: int | None = None
    blk: int | None = None
    tov: int | None = None
    pf: int | None = None
    plus_minus: float | None = None


@dataclass(frozen=True)
class LivePlay:
    provider_play_id: str
    sequence: int
    period: int
    clock_s: float | None
    wallclock: datetime | None
    play_type: str
    text: str
    team: str | None
    scoring: bool
    score_value: int
    away_score: int | None
    home_score: int | None
    provider_player_ids: tuple[str, ...]


@dataclass(frozen=True)
class LiveGameState:
    provider: str
    provider_game_id: str
    fetched_at: datetime
    game_date: date
    away: str
    home: str
    status: str                  # pre | in | post
    period: int | None
    clock_s: float | None
    away_score: int | None
    home_score: int | None
    home_spread: float | None
    total: float | None
    players: list[LivePlayerLine] = field(default_factory=list)
    plays: list[LivePlay] = field(default_factory=list)
    is_mock: bool = False

    @property
    def newest_play_at(self) -> datetime | None:
        stamps = [p.wallclock for p in self.plays if p.wallclock is not None]
        return max(stamps) if stamps else None

    @property
    def feed_lag_s(self) -> float | None:
        """How old the newest play was when we fetched. During live play this is our feed delay
        (plus any dead-ball time, so it's an upper bound)."""
        newest = self.newest_play_at
        return None if newest is None else (self.fetched_at - newest).total_seconds()


@dataclass(frozen=True)
class InjuryEntry:
    provider_player_id: str | None
    name: str
    team_name: str
    status: str                  # e.g. "Out", "Day-To-Day", "Questionable"
    comment: str
    updated: datetime | None


class LiveStatsProvider(ABC):
    name: str
    is_mock: bool = False

    @abstractmethod
    def scoreboard(self, day: date | None = None) -> list[ScheduledGame] | None:
        """Games on a US-Eastern date (default: the provider's 'today'). [] = no games;
        None = the fetch failed (callers must keep their previous state, not assume no games)."""

    @abstractmethod
    def game_state(self, provider_game_id: str) -> LiveGameState | None:
        """Full current state of one game, or None if the fetch failed."""

    def injuries(self) -> list[InjuryEntry]:
        """Current injury report (providers without one return an empty list)."""
        return []
