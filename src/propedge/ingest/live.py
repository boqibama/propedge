"""Live poller: during games, store ESPN game state (box score, play-by-play, feed lag) and
Kalshi quotes into the database, each with the UTC time we saw it.

One loop, three schedules (all from config.yaml):
- scoreboard every `live_feed.scoreboard_interval_s`  -> which games are live
- each live game every `live_feed.poll_interval_s`    -> snapshot + new plays
- odds every `odds.poll_interval_s`                    -> quotes (a new row only when a quote changes)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import astuple
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from propedge.data.live_base import LiveGameState, LiveStatsProvider
from propedge.db.models import (LiveGameSnapshot, LivePlayerSnapshot, OddsSnapshot, Play, PropMarket)
from propedge.db.store import PlayerResolver, ensure_game, upsert
from propedge.ids import game_key
from propedge.odds.base import OddsProvider, PropQuote

log = logging.getLogger(__name__)

_QUOTE_FIELDS = ("yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size", "last_price", "volume", "open_interest")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LiveStore:
    """Writes provider data into the database. Keeps small in-memory caches so repeated polls
    only write what changed."""

    def __init__(self, session: Session, stale_after_s: float, stale_hard_after_s: float | None = None,
                 is_mock: bool = False):
        self.session = session
        self.stale_after_s = stale_after_s
        self.stale_hard_after_s = stale_hard_after_s if stale_hard_after_s is not None else 4 * stale_after_s
        self._last_clock: dict[str, tuple] = {}                # game_id -> (period, clock_s) last snapshot
        self.is_mock = is_mock
        self.resolver = PlayerResolver(session)
        self._seen_plays: dict[str, set[str]] = {}            # game_id -> provider play ids stored
        self._box_hash: dict[str, str] = {}                    # game_id -> hash of last stored box score
        self._quotes: dict[str, tuple[int, tuple]] = {}        # ticker -> (snapshot row id, quote values)
        self._load_latest_quotes()

    # ------------------------------------------------------------------ games

    def store_game_state(self, state: LiveGameState) -> LiveGameSnapshot:
        s = self.session
        gid = ensure_game(s, state.game_date, state.away, state.home,
                          espn_event_id=state.provider_game_id if state.provider == "espn" else None,
                          status=state.status, away_score=state.away_score, home_score=state.home_score,
                          home_spread=state.home_spread, total=state.total, updated_at=state.fetched_at)

        espn_to_player = {pl.provider_player_id: self.resolver.from_espn(pl.provider_player_id, pl.name,
                                                                          pl.team, pl.position)
                          for pl in state.players}

        new_plays = self._store_new_plays(gid, state, espn_to_player)
        lag = state.feed_lag_s
        delays = sorted((state.fetched_at - p.wallclock).total_seconds() for p, backlog in new_plays
                        if not backlog and p.wallclock is not None)
        clock = (state.period, state.clock_s)
        prev_clock = self._last_clock.get(gid)
        self._last_clock[gid] = clock
        clock_running = None if prev_clock is None else clock != prev_clock
        snap = LiveGameSnapshot(
            game_id=gid, provider=state.provider, fetched_at=state.fetched_at, status=state.status,
            period=state.period, clock_s=state.clock_s, away_score=state.away_score,
            home_score=state.home_score, n_plays=len(state.plays), newest_play_at=state.newest_play_at,
            feed_lag_s=lag, play_delay_s=delays[len(delays) // 2] if delays else None,
            clock_running=clock_running,
            is_stale=self._is_stale(state.status, lag, clock_running),
            is_mock=self.is_mock or state.is_mock)
        s.add(snap)
        s.flush()

        # Player rows only when the box score actually changed (saves ~90% of rows).
        box_hash = hashlib.sha1(repr(sorted(astuple(p) for p in state.players)).encode()).hexdigest()
        if self._box_hash.get(gid) != box_hash:
            self._box_hash[gid] = box_hash
            s.add_all(LivePlayerSnapshot(
                snapshot_id=snap.id, player_id=espn_to_player[p.provider_player_id], team=p.team,
                starter=p.starter, did_not_play=p.did_not_play, minutes=p.minutes, pts=p.pts, fgm=p.fgm,
                fga=p.fga, fg3m=p.fg3m, fg3a=p.fg3a, ftm=p.ftm, fta=p.fta, reb=p.reb, ast=p.ast,
                stl=p.stl, blk=p.blk, tov=p.tov, pf=p.pf, plus_minus=p.plus_minus) for p in state.players)

        s.commit()
        return snap

    def _is_stale(self, status: str, lag: float | None, clock_running: bool | None) -> bool:
        """Lagging = the game clock is moving but no new plays reach us for `stale_after_s`.
        If the clock is stopped (timeout, review, free throws) an old newest-play is normal —
        unless it exceeds `stale_hard_after_s`, in case the whole feed (clock included) froze."""
        if status != "in" or lag is None:
            return False
        if lag > self.stale_hard_after_s:
            return True
        return lag > self.stale_after_s and clock_running is not False

    def _store_new_plays(self, gid: str, state: LiveGameState,
                         espn_to_player: dict[str, int]) -> list[tuple]:
        """Insert plays we haven't stored yet. Returns [(play, is_backlog)] for the new ones."""
        seen = self._seen_plays.get(gid)
        backlog = seen is None          # first look at this game in this process
        if seen is None:
            seen = set(self.session.scalars(select(Play.provider_play_id).where(
                Play.game_id == gid, Play.provider == state.provider)))
            self._seen_plays[gid] = seen
        new = [p for p in state.plays if p.provider_play_id not in seen]
        for p in new:
            self.session.add(Play(
                provider=state.provider, provider_play_id=p.provider_play_id, game_id=gid,
                sequence=p.sequence, period=p.period, clock_s=p.clock_s, wallclock=p.wallclock,
                first_seen_at=state.fetched_at, play_type=p.play_type, text=p.text, team=p.team,
                scoring=p.scoring, score_value=p.score_value, away_score=p.away_score,
                home_score=p.home_score,
                player_ids=[espn_to_player[x] for x in p.provider_player_ids if x in espn_to_player],
                backlog=backlog, is_mock=self.is_mock or state.is_mock))
            seen.add(p.provider_play_id)
        return [(p, backlog) for p in new]

    # ------------------------------------------------------------------ odds

    def _load_latest_quotes(self) -> None:
        latest = select(func.max(OddsSnapshot.id)).group_by(OddsSnapshot.ticker).scalar_subquery()
        for row in self.session.scalars(select(OddsSnapshot).where(OddsSnapshot.id.in_(latest))):
            self._quotes[row.ticker] = (row.id, tuple(getattr(row, f) for f in _QUOTE_FIELDS))

    def store_quotes(self, quotes: list[PropQuote]) -> dict[str, int]:
        """Upsert market metadata; write a snapshot row only for quotes that changed, and just
        bump `last_seen_at` for the rest."""
        s = self.session
        markets = []
        for q in quotes:
            gid = game_key(q.game_date, q.away, q.home) if q.game_date and q.away and q.home else None
            markets.append(dict(ticker=q.ticker, provider=q.provider, event_ticker=q.event_ticker, game_id=gid,
                                player_id=self.resolver.from_name(q.player_name, q.team),
                                player_name=q.player_name, team=q.team, stat=q.stat, line=q.line,
                                close_time=q.close_time, status=q.status, is_mock=self.is_mock or q.is_mock))
        upsert(s, PropMarket, markets, ["ticker"])

        changed, unchanged_ids = 0, []
        new_rows: list[tuple[str, OddsSnapshot]] = []
        for q in quotes:
            values = tuple(getattr(q, f) for f in _QUOTE_FIELDS)
            prev = self._quotes.get(q.ticker)
            if prev and prev[1] == values:
                unchanged_ids.append(prev[0])
                continue
            row = OddsSnapshot(ticker=q.ticker, first_seen_at=q.fetched_at, last_seen_at=q.fetched_at,
                               is_mock=self.is_mock or q.is_mock,
                               **dict(zip(_QUOTE_FIELDS, values)))
            s.add(row)
            new_rows.append((q.ticker, row))
            changed += 1
        if unchanged_ids and quotes:
            s.execute(update(OddsSnapshot).where(OddsSnapshot.id.in_(unchanged_ids))
                      .values(last_seen_at=quotes[0].fetched_at))
        s.flush()
        for ticker, row in new_rows:
            self._quotes[ticker] = (row.id, tuple(getattr(row, f) for f in _QUOTE_FIELDS))
        s.commit()
        return {"markets": len(markets), "changed": changed, "unchanged": len(unchanged_ids)}


class LivePoller:
    def __init__(self, session_factory: sessionmaker[Session], live: LiveStatsProvider,
                 odds: OddsProvider | None, config: dict, stats: list[str],
                 now: Callable[[], datetime] = _utcnow, sleep: Callable[[float], None] = time.sleep):
        lf, oc = config["live_feed"], config["odds"]
        self.live, self.odds, self.stats = live, odds, stats
        self.scoreboard_every = timedelta(seconds=lf["scoreboard_interval_s"])
        self.game_every = timedelta(seconds=lf["poll_interval_s"])
        self.odds_every = timedelta(seconds=oc["poll_interval_s"])
        self.odds_every_live = timedelta(seconds=oc["live_poll_interval_s"])   # faster while games are live
        self._closing_done: set[str] = set()
        self.now, self.sleep = now, sleep
        is_mock = live.is_mock or bool(odds and odds.is_mock)
        self.store = LiveStore(session_factory(), stale_after_s=lf["stale_after_s"],
                               stale_hard_after_s=lf["stale_hard_after_s"], is_mock=is_mock)
        self.live_games: list[str] = []            # provider game ids currently in progress
        self.finishing: list[str] = []             # just went final: fetch once more to store the final box
        self._next_scoreboard = self._next_odds = datetime.min.replace(tzinfo=timezone.utc)
        self._next_game: dict[str, datetime] = {}
        self.all_final = False
        self._last_tick: datetime | None = None
        self.gap_warn_s = lf["gap_warn_s"]
        self.scoreboard_retry = timedelta(seconds=lf["scoreboard_retry_s"])
        self.engine = None          # optional LiveEngine (paper predictions + flags during games)
        self.day = None             # fixed US-Eastern game day (tonight); None = the provider's default "today"
        self.ticks = 0
        # Extra periodic jobs, e.g. the pre-game Kalshi comparison: [(interval, fn(now)), ...]
        self.jobs: list[tuple[timedelta, Callable[[datetime], None]]] = []
        self._next_job: dict[int, datetime] = {}

    def tick(self) -> None:
        """Do whatever is due right now."""
        now = self.now()
        if self._last_tick is not None and (now - self._last_tick).total_seconds() > self.gap_warn_s:
            log.warning("poller: %.0f s since the last tick — the computer may have slept or the network "
                        "dropped; data has a gap", (now - self._last_tick).total_seconds())
        self._last_tick = now
        if now >= self._next_scoreboard:
            games = self.live.scoreboard(self.day) if self.day else self.live.scoreboard()
            if games is None:       # fetch failed: keep what we knew, try again soon
                self._next_scoreboard = now + self.scoreboard_retry
                games = []
                failed = True
            else:
                failed = False
            for g in games:
                ensure_game(self.store.session, g.game_date, g.away, g.home,
                            espn_event_id=g.provider_game_id if self.live.name == "espn" else None,
                            start_time=g.start_time, status=g.status, away_score=g.away_score,
                            home_spread=g.home_spread, total=g.total,
                            home_score=g.home_score, updated_at=now)
            self.store.session.commit()
            if not failed:
                now_live = [g.provider_game_id for g in games if g.status == "in"]
                self.finishing += [gid for gid in self.live_games if gid not in now_live]
                self.live_games = now_live
                # "All final" needs games that exist and are all over — an empty day counts too, but
                # only when the fetch succeeded (a failed fetch is handled above).
                self.all_final = all(g.status == "post" for g in games)
                self._next_scoreboard = now + self.scoreboard_every
                log.info("scoreboard: %d games, %d live", len(games), len(self.live_games))

        finishing, self.finishing = self.finishing, []
        for gid in self.live_games + finishing:
            if gid in finishing or now >= self._next_game.get(gid, now):
                state = self.live.game_state(gid)
                if state is not None:
                    snap = self.store.store_game_state(state)
                    if self.engine is not None:
                        try:
                            self.engine.on_state(state, snap.game_id, snap.fetched_at, snap.is_stale)
                        except Exception:   # the paper engine must never stop data collection
                            log.exception("live engine failed for %s", snap.game_id)
                    if state.status == "in" and snap.game_id not in self._closing_done:
                        self._capture_closing(snap.game_id)
                    fmt = lambda x: f"{x:.0f}s" if x is not None else "-"
                    log.info("%s %s-%s Q%s %s plays=%d newest_play_age=%s play_delay=%s%s%s", snap.game_id,
                             state.away_score, state.home_score, state.period, state.clock_s, snap.n_plays,
                             fmt(snap.feed_lag_s), fmt(snap.play_delay_s),
                             "" if snap.clock_running is not False else " (clock stopped)",
                             " STALE" if snap.is_stale else "")
                self._next_game[gid] = now + self.game_every

        if self.odds and now >= self._next_odds:
            quotes = self.odds.fetch_quotes(self.stats)
            result = self.store.store_quotes(quotes)
            log.info("odds: %d markets, %d changed", result["markets"], result["changed"])
            if self.store.resolver.unmatched_names:
                log.debug("unmatched odds names: %s", sorted(self.store.resolver.unmatched_names))
            self._next_odds = now + (self.odds_every_live if self.live_games else self.odds_every)

        for i, (every, job) in enumerate(self.jobs):
            if now >= self._next_job.get(i, now):
                try:
                    job(now)
                except Exception:   # a failing side job must never stop data collection
                    log.exception("periodic job %s failed", getattr(job, "__name__", job))
                self._next_job[i] = now + every
        self.ticks += 1

    def _capture_closing(self, game_id: str) -> None:
        """Freeze Kalshi's last pre-tip quote for every contract of a game that just went live."""
        from propedge.ingest.closing import capture_from_snapshots, tip_time

        tip = tip_time(self.store.session, game_id)
        if tip is None:
            return
        n = capture_from_snapshots(self.store.session, game_id, tip)
        self._closing_done.add(game_id)
        log.info("closing quotes: %s tipped off at %s, saved %d contracts", game_id, tip.strftime("%H:%M:%S"), n)

    def run(self, max_minutes: float | None = None, stop_when_idle: bool = False,
            stop_when_all_final: bool = False) -> None:
        """stop_when_idle: stop as soon as no game is live (also before any game starts).
        stop_when_all_final: keep going through pre-game, stop once every game today is over."""
        start = self.now()
        while True:
            self.tick()
            if max_minutes is not None and self.now() - start >= timedelta(minutes=max_minutes):
                log.info("poller: reached --minutes %s, stopping", max_minutes)
                return
            if stop_when_idle and not self.live_games and self.ticks > 1:
                log.info("poller: no live games, stopping")
                return
            if stop_when_all_final and self.all_final and self.ticks > 1:
                log.info("poller: all of today's games are final, stopping")
                return
            self.sleep(1.0)
