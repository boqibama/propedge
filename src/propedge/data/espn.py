"""ESPN implementation of LiveStatsProvider (public, unofficial JSON — no key).

The parse_* functions are pure (dict in, dataclasses out) so they can be tested against
saved payloads without the network.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Callable

from propedge.data.live_base import (InjuryEntry, LiveGameState, LivePlay, LivePlayerLine,
                                     LiveStatsProvider, ScheduledGame)
from propedge.http import HttpClient, make_client
from propedge.ids import clock_to_seconds, eastern_date, minutes_to_float, tricode_from_espn

log = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _made_attempted(value: str | None) -> tuple[int | None, int | None]:
    """'7-15' -> (7, 15)."""
    if not value or "-" not in value:
        return None, None
    made, att = value.split("-", 1)
    return _int(made), _int(att)


def _state(status: dict) -> str:
    return status.get("type", {}).get("state", "pre")   # pre | in | post


# ----------------------------------------------------------------------------- scoreboard

def parse_scoreboard(payload: dict) -> list[ScheduledGame]:
    games = []
    for event in payload.get("events", []):
        comp = event["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        start = _ts(event.get("date"))
        odds = (comp.get("odds") or [{}])[0]
        games.append(ScheduledGame(
            provider_game_id=str(event["id"]),
            game_date=eastern_date(start) if start else date.today(),
            start_time=start,
            away=tricode_from_espn(teams["away"]["team"]["abbreviation"]),
            home=tricode_from_espn(teams["home"]["team"]["abbreviation"]),
            status=_state(event.get("status", {})),
            away_score=_int(teams["away"].get("score")),
            home_score=_int(teams["home"].get("score")),
            home_spread=odds.get("spread"),           # same convention as the summary's pickcenter
            total=odds.get("overUnder"),
        ))
    return games


# ----------------------------------------------------------------------------- game summary

def _parse_players(boxscore: dict) -> list[LivePlayerLine]:
    lines = []
    for team_block in boxscore.get("players", []):
        team = tricode_from_espn(team_block["team"]["abbreviation"])
        for group in team_block.get("statistics", []):
            keys = group.get("keys", [])
            for row in group.get("athletes", []):
                athlete = row.get("athlete") or {}
                if not athlete.get("id"):      # occasional placeholder rows without an athlete
                    continue
                stats = dict(zip(keys, row.get("stats") or []))
                fgm, fga = _made_attempted(stats.get("fieldGoalsMade-fieldGoalsAttempted"))
                fg3m, fg3a = _made_attempted(stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                ftm, fta = _made_attempted(stats.get("freeThrowsMade-freeThrowsAttempted"))
                pm = stats.get("plusMinus")
                lines.append(LivePlayerLine(
                    provider_player_id=str(athlete["id"]),
                    name=athlete["displayName"],
                    team=team,
                    position=(athlete.get("position") or {}).get("abbreviation"),
                    starter=row.get("starter"),
                    did_not_play=bool(row.get("didNotPlay")),
                    minutes=minutes_to_float(stats.get("minutes")),
                    pts=_int(stats.get("points")),
                    fgm=fgm, fga=fga, fg3m=fg3m, fg3a=fg3a, ftm=ftm, fta=fta,
                    reb=_int(stats.get("rebounds")),
                    ast=_int(stats.get("assists")),
                    stl=_int(stats.get("steals")),
                    blk=_int(stats.get("blocks")),
                    tov=_int(stats.get("turnovers")),
                    pf=_int(stats.get("fouls")),
                    plus_minus=float(pm) if pm not in (None, "", "--") else None,
                ))
    return lines


def _parse_plays(raw_plays: list[dict], team_by_espn_id: dict[str, str]) -> list[LivePlay]:
    plays = []
    for p in raw_plays:
        plays.append(LivePlay(
            provider_play_id=str(p["id"]),
            sequence=_int(p.get("sequenceNumber")) or 0,
            period=(p.get("period") or {}).get("number", 0),
            clock_s=clock_to_seconds((p.get("clock") or {}).get("displayValue")),
            wallclock=_ts(p.get("wallclock")),
            play_type=(p.get("type") or {}).get("text", "").replace("\n", " "),
            text=p.get("text", ""),
            team=team_by_espn_id.get(str((p.get("team") or {}).get("id"))),
            scoring=bool(p.get("scoringPlay")),
            score_value=_int(p.get("scoreValue")) or 0,
            away_score=_int(p.get("awayScore")),
            home_score=_int(p.get("homeScore")),
            provider_player_ids=tuple(str(x["athlete"]["id"]) for x in p.get("participants") or []
                                      if x.get("athlete", {}).get("id")),
        ))
    return plays


def parse_summary(payload: dict, fetched_at: datetime, provider: str = "espn") -> LiveGameState:
    comp = payload["header"]["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    team_by_espn_id = {str(c["team"]["id"]): tricode_from_espn(c["team"]["abbreviation"])
                       for c in comp["competitors"]}
    status = comp.get("status", {})
    start = _ts(comp.get("date"))
    pick = (payload.get("pickcenter") or [{}])[0]
    return LiveGameState(
        provider=provider,
        provider_game_id=str(payload["header"]["id"]),
        fetched_at=fetched_at,
        game_date=eastern_date(start) if start else eastern_date(fetched_at),
        away=tricode_from_espn(teams["away"]["team"]["abbreviation"]),
        home=tricode_from_espn(teams["home"]["team"]["abbreviation"]),
        status=_state(status),
        period=status.get("period"),
        clock_s=clock_to_seconds(status.get("displayClock")),
        away_score=_int(teams["away"].get("score")),
        home_score=_int(teams["home"].get("score")),
        home_spread=pick.get("spread"),          # ESPN's `spread` is the home line (−6.5 = home favoured)
        total=pick.get("overUnder"),
        players=_parse_players(payload.get("boxscore", {})),
        plays=_parse_plays(payload.get("plays", []), team_by_espn_id),
    )


# ----------------------------------------------------------------------------- injuries

def parse_injuries(payload: dict) -> list[InjuryEntry]:
    out = []
    for team in payload.get("injuries", []):
        for i in team.get("injuries", []):
            a = i.get("athlete") or {}
            athlete_id = None
            for link in a.get("links", []):      # ESPN puts the athlete id only in profile URLs
                m = re.search(r"/id/(\d+)/", link.get("href", ""))
                if m:
                    athlete_id = m.group(1)
                    break
            out.append(InjuryEntry(provider_player_id=athlete_id, name=a.get("displayName", ""),
                                   team_name=team.get("displayName", ""), status=i.get("status", ""),
                                   comment=i.get("shortComment", ""), updated=_ts(i.get("date"))))
    return out


# ----------------------------------------------------------------------------- provider

class EspnLiveProvider(LiveStatsProvider):
    name = "espn"

    def __init__(self, client: HttpClient | None = None,
                 on_raw: Callable[[str, str, datetime, dict], None] | None = None):
        self.client = client or make_client("espn")
        self.on_raw = on_raw          # e.g. a RawRecorder, to save payloads for replay

    def scoreboard(self, day: date | None = None) -> list[ScheduledGame] | None:
        """Games on a day, [] if there are none — or None if the fetch FAILED (never treat that as "no games")."""
        # limit: 2026-10-27 ESPN served an empty 200 for exactly "?dates=YYYYMMDD" (a stuck edge cache) while
        # the same request with &limit=100 worked. A day never has more than ~15 games.
        params = {"dates": day.strftime("%Y%m%d"), "limit": 100} if day else None
        r = self.client.get_json(f"{BASE_URL}/scoreboard", params=params)
        if not r.ok:
            log.warning("ESPN scoreboard failed: %s", r.error)
            return None
        return parse_scoreboard(r.data)

    def injuries(self) -> list[InjuryEntry]:
        r = self.client.get_json(f"{BASE_URL}/injuries")
        if not r.ok:
            log.warning("ESPN injuries failed: %s", r.error)
            return []
        if self.on_raw:
            self.on_raw("espninjuries", "all", r.fetched_at, r.data)
        return parse_injuries(r.data)

    def game_state(self, provider_game_id: str) -> LiveGameState | None:
        r = self.client.get_json(f"{BASE_URL}/summary", params={"event": provider_game_id})
        if not r.ok:
            log.warning("ESPN summary %s failed: %s", provider_game_id, r.error)
            return None
        if self.on_raw:
            self.on_raw("espn", provider_game_id, r.fetched_at, r.data)
        try:
            return parse_summary(r.data, r.fetched_at.astimezone(timezone.utc))
        except (KeyError, TypeError, ValueError, IndexError) as exc:   # unofficial API: never crash the poller
            log.warning("ESPN summary %s could not be parsed: %s: %s", provider_game_id, type(exc).__name__, exc)
            return None
