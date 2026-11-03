"""Historical backfill: game logs, team stats, matchups (stats.nba.com) and the schedule (ESPN).

Every step is idempotent: re-running updates rows in place, so `backfill` can run daily.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from nba_api.stats.endpoints import boxscorematchupsv3, leaguedashteamstats, leaguegamelog
from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.data.live_base import LiveStatsProvider
from propedge.data.nba_stats import FOREVER, NbaStatsClient, NbaStatsError, result_set
from propedge.db.models import Game, Matchup, PlayerGameLog, Team, TeamGameLog, TeamSeasonStats
from propedge.db.models import Play as m_plays
from propedge.db.store import PlayerResolver, ensure_game, ensure_team, upsert
from propedge.config import load_config, resolve_path
from propedge.ids import EASTERN, season_for

log = logging.getLogger(__name__)


def _is_current(season: str) -> bool:
    return season == season_for(datetime.now(EASTERN).date())


def _team_log_rows(client: NbaStatsClient, season: str, refresh: bool) -> list[dict]:
    ttl = "default" if _is_current(season) else FOREVER
    return result_set(client.call(leaguegamelog.LeagueGameLog, cache_ttl_s=ttl, refresh=refresh,
                                  season=season, player_or_team_abbreviation="T",
                                  season_type_all_star="Regular Season"))


def game_sides(team_rows: list[dict]) -> dict[str, tuple[str, str, bool]]:
    """GAME_ID -> (away, home, neutral_site), decided once per game from both teams' rows.

    MATCHUP is 'LAL vs. GSW' for the home team and 'GSW @ LAL' for the visitor. At neutral
    sites (NBA Cup in Las Vegas, international games) BOTH rows say '@'; then we keep the
    orientation of the alphabetically-first team's row and mark the game neutral.
    """
    by_game: dict[str, list[dict]] = {}
    for r in team_rows:
        by_game.setdefault(r["GAME_ID"], []).append(r)
    sides = {}
    for gid, rows in by_game.items():
        home_rows = [r for r in rows if " vs. " in r["MATCHUP"]]
        if len(home_rows) == 1:
            home = home_rows[0]["TEAM_ABBREVIATION"]
            away = home_rows[0]["MATCHUP"].split(" vs. ")[1]
            sides[gid] = (away, home, False)
        else:
            first = min(rows, key=lambda r: r["TEAM_ABBREVIATION"])
            away, home = first["MATCHUP"].split(" @ ") if " @ " in first["MATCHUP"] else first["MATCHUP"].split(" vs. ")[::-1]
            sides[gid] = (away, home, True)
    return sides


def _home_away(row: dict, sides: dict[str, tuple[str, str, bool]]) -> tuple[bool, str, str, str]:
    """(is_home, opponent, away, home) for one team's row. Neutral-site games: is_home is False for both."""
    away, home, neutral = sides[row["GAME_ID"]]
    team = row["TEAM_ABBREVIATION"]
    opp = home if team == away else away
    return (team == home and not neutral), opp, away, home


def ingest_team_logs(session: Session, client: NbaStatsClient, season: str, refresh: bool = False) -> int:
    rows = _team_log_rows(client, season, refresh)
    sides = game_sides(rows)
    out = []
    for r in rows:
        is_home, opp, away, home = _home_away(r, sides)
        gdate = date.fromisoformat(r["GAME_DATE"][:10])
        ensure_team(session, r["TEAM_ABBREVIATION"], nba_team_id=r["TEAM_ID"], name=r["TEAM_NAME"])
        gid = ensure_game(session, gdate, away, home, nba_game_id=r["GAME_ID"], status="post")
        out.append(dict(nba_game_id=r["GAME_ID"], game_id=gid, season=season, game_date=gdate,
                        team=r["TEAM_ABBREVIATION"], opponent=opp, is_home=is_home, minutes=r["MIN"],
                        pts=r["PTS"], fga=r["FGA"], fta=r["FTA"], oreb=r["OREB"], tov=r["TOV"], pf=r["PF"]))
    session.flush()
    return upsert(session, TeamGameLog, out, ["nba_game_id", "team"])


def ingest_player_logs(session: Session, client: NbaStatsClient, season: str, refresh: bool = False) -> int:
    ttl = "default" if _is_current(season) else FOREVER
    rows = result_set(client.call(leaguegamelog.LeagueGameLog, cache_ttl_s=ttl, refresh=refresh,
                                  season=season, player_or_team_abbreviation="P",
                                  season_type_all_star="Regular Season"))
    sides = game_sides(_team_log_rows(client, season, refresh=False))
    resolver = PlayerResolver(session)
    # Rows come newest-first; process oldest-first so each player's `team` ends as their latest.
    rows.sort(key=lambda r: r["GAME_DATE"])
    out = []
    for r in rows:
        is_home, opp, away, home = _home_away(r, sides)
        gdate = date.fromisoformat(r["GAME_DATE"][:10])
        pid = resolver.from_nba(r["PLAYER_ID"], r["PLAYER_NAME"], r["TEAM_ABBREVIATION"])
        gid = ensure_game(session, gdate, away, home, nba_game_id=r["GAME_ID"], status="post")
        out.append(dict(nba_game_id=r["GAME_ID"], nba_player_id=r["PLAYER_ID"], player_id=pid, game_id=gid,
                        season=season, game_date=gdate, team=r["TEAM_ABBREVIATION"], opponent=opp,
                        is_home=is_home, minutes=r["MIN"], pts=r["PTS"], fgm=r["FGM"], fga=r["FGA"],
                        fg3m=r["FG3M"], fg3a=r["FG3A"], ftm=r["FTM"], fta=r["FTA"], oreb=r["OREB"],
                        dreb=r["DREB"], reb=r["REB"], ast=r["AST"], stl=r["STL"], blk=r["BLK"],
                        tov=r["TOV"], pf=r["PF"], plus_minus=r["PLUS_MINUS"]))
    session.flush()
    return upsert(session, PlayerGameLog, out, ["nba_game_id", "nba_player_id"])


def ingest_team_advanced(session: Session, client: NbaStatsClient, season: str, refresh: bool = False) -> int:
    ttl = "default" if _is_current(season) else FOREVER
    rows = result_set(client.call(leaguedashteamstats.LeagueDashTeamStats, cache_ttl_s=ttl, refresh=refresh,
                                  season=season, measure_type_detailed_defense="Advanced"))
    tricode_by_id = {t.nba_team_id: t.tricode for t in session.scalars(select(Team))}
    as_of = datetime.now(EASTERN).date() if _is_current(season) else date(int(season[:4]) + 1, 6, 30)
    out = [dict(season=season, team=tricode_by_id[r["TEAM_ID"]], as_of=as_of, games=r["GP"], pace=r["PACE"],
                off_rating=r["OFF_RATING"], def_rating=r["DEF_RATING"])
           for r in rows if r["TEAM_ID"] in tricode_by_id]
    return upsert(session, TeamSeasonStats, out, ["season", "team", "as_of"])


def ingest_matchups(session: Session, client: NbaStatsClient, season: str, limit: int | None = None) -> int:
    """BoxScoreMatchupsV3 for each finished game of the season we don't have yet (1 request per game)."""
    have = set(session.scalars(select(Matchup.nba_game_id).distinct()))
    todo = [g for g in session.scalars(select(Game.nba_game_id).where(
        Game.season == season, Game.nba_game_id.is_not(None), Game.status == "post").order_by(Game.game_date))
            if g not in have]
    if limit:
        todo = todo[:limit]
    log.info("matchups: %d games to fetch for %s", len(todo), season)
    n = 0
    for i, nba_game_id in enumerate(todo, 1):
        try:
            data = client.call(boxscorematchupsv3.BoxScoreMatchupsV3, cache_ttl_s=FOREVER, game_id=nba_game_id)
        except NbaStatsError as exc:
            log.warning("matchups for %s failed: %s", nba_game_id, exc)
            continue
        except (IndexError, KeyError) as exc:
            # nba_api's parser crashes on an empty matchup payload (game not processed yet);
            # nothing is cached, so the next backfill tries again.
            log.info("matchups for %s not available yet (%s)", nba_game_id, type(exc).__name__)
            continue
        out = []
        bm = data.get("boxScoreMatchups") or {}
        for side in ("homeTeam", "awayTeam"):
            for off in (bm.get(side) or {}).get("players", []):
                for m in off.get("matchups", []):
                    s = m["statistics"]
                    out.append(dict(nba_game_id=nba_game_id, off_nba_player_id=off["personId"],
                                    def_nba_player_id=m["personId"],
                                    matchup_seconds=s.get("matchupMinutesSort"),
                                    partial_possessions=s.get("partialPossessions"),
                                    off_points=s.get("playerPoints"),
                                    off_fga=s.get("matchupFieldGoalsAttempted"),
                                    off_fgm=s.get("matchupFieldGoalsMade"),
                                    off_fg3a=s.get("matchupThreePointersAttempted")))
        n += upsert(session, Matchup, out, ["nba_game_id", "off_nba_player_id", "def_nba_player_id"])
        if i % 25 == 0:
            session.commit()
            log.info("matchups: %d/%d games", i, len(todo))
    return n


def ingest_schedule(session: Session, provider: LiveStatsProvider, start: date, end: date) -> int:
    """Upcoming/recent games from the live provider's scoreboard (for rest days, spreads, ids)."""
    n = 0
    day = start
    while day <= end:
        for g in provider.scoreboard(day) or []:
            ensure_game(session, g.game_date, g.away, g.home, espn_event_id=g.provider_game_id,
                        start_time=g.start_time, status=g.status, away_score=g.away_score,
                            home_spread=g.home_spread, total=g.total,
                        home_score=g.home_score, updated_at=datetime.now(timezone.utc))
            n += 1
        day += timedelta(days=1)
    return n


def ingest_historical_spreads(session: Session, provider: LiveStatsProvider, season: str) -> int:
    """Closing spread/total for each finished game of a season (from the provider's per-game summary,
    one request per game). Only fills games that don't have a spread yet, so re-runs are cheap.
    Updates existing games only — it never creates games (neutral-site orientation can differ)."""
    # Games ESPN has no line for are remembered, so daily runs don't re-download their summaries.
    no_line_file = resolve_path(load_config()["paths"]["data_dir"]) / "cache" / "games_without_spread.json"
    no_line = set(json.loads(no_line_file.read_text())) if no_line_file.exists() else set()
    missing = {g.id for g in session.scalars(select(Game).where(
        Game.season == season, Game.status == "post", Game.home_spread.is_(None)))} - no_line
    if not missing:
        return 0
    dates = sorted({session.get(Game, gid).game_date for gid in missing})
    log.info("spreads: %d games missing for %s across %d dates (1 request per game)", len(missing), season, len(dates))
    filled = 0
    for i, day in enumerate(dates, 1):
        for g in provider.scoreboard(day) or []:
            key, flip = f"{g.game_date.isoformat()}_{g.away}_{g.home}", False
            if key not in missing:
                key, flip = f"{g.game_date.isoformat()}_{g.home}_{g.away}", True
            if key not in missing:
                continue
            state = provider.game_state(g.provider_game_id)
            if state is not None and state.home_spread is None:
                no_line.add(key)
            if state is None or state.home_spread is None:
                continue
            game = session.get(Game, key)
            game.home_spread = -state.home_spread if flip else state.home_spread
            game.total = state.total
            if game.espn_event_id is None:
                game.espn_event_id = g.provider_game_id
            missing.discard(key)
            filled += 1
        session.commit()            # short write transactions: other processes can write in between
        no_line_file.parent.mkdir(parents=True, exist_ok=True)
        no_line_file.write_text(json.dumps(sorted(no_line)))
        if i % 10 == 0:
            log.info("spreads: %d/%d dates, %d games filled", i, len(dates), filled)
    session.commit()
    return filled


def ingest_playbyplay(session: Session, provider: LiveStatsProvider, start: date, end: date,
                      skip_existing: bool = True) -> int:
    """Full play-by-play + final box score for every finished game between `start` and `end`
    (one summary request per game), stored like a live poll (plays marked as backlog).
    Used to rebuild in-game states for the live-engine backtest and the repricing test."""
    from propedge.ingest.live import LiveStore

    store = LiveStore(session, stale_after_s=load_config()["live_feed"]["stale_after_s"])
    have = set(session.scalars(select(Game.espn_event_id).where(Game.id.in_(
        select(m_plays.game_id).distinct())))) if skip_existing else set()
    n, day = 0, start
    while day <= end:
        for g in provider.scoreboard(day) or []:
            if g.status != "post" or g.provider_game_id in have:
                continue
            state = provider.game_state(g.provider_game_id)
            if state is None or not state.plays:
                continue
            store.store_game_state(state)
            n += 1
        log.info("play-by-play: %s done (%d games so far)", day, n)
        day += timedelta(days=1)
    return n


def backfill(session: Session, client: NbaStatsClient, provider: LiveStatsProvider, seasons: list[str],
             matchup_seasons: list[str], schedule_days_back: int, schedule_days_ahead: int,
             spread_seasons: list[str] = (), refresh: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    for season in seasons:
        counts[f"team_logs {season}"] = ingest_team_logs(session, client, season, refresh)
        session.commit()
        counts[f"player_logs {season}"] = ingest_player_logs(session, client, season, refresh)
        session.commit()
        counts[f"team_advanced {season}"] = ingest_team_advanced(session, client, season, refresh)
        session.commit()
    today = datetime.now(EASTERN).date()
    counts["schedule"] = ingest_schedule(session, provider, today - timedelta(days=schedule_days_back),
                                         today + timedelta(days=schedule_days_ahead))
    session.commit()
    for season in matchup_seasons:
        counts[f"matchups {season}"] = ingest_matchups(session, client, season)
        session.commit()
    for season in spread_seasons:
        counts[f"spreads {season}"] = ingest_historical_spreads(session, provider, season)
    return counts
