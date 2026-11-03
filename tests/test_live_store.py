"""LiveStore / LivePoller against a temporary SQLite database, with fake providers."""

import gzip
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from propedge.config import load_config
from propedge.data.espn import parse_summary
from propedge.data.live_base import LiveStatsProvider, ScheduledGame
from propedge.data.replay import ReplayClock, ReplayLiveProvider, ReplayOddsProvider
from propedge.db import models as m
from propedge.db.session import init_db, make_engine, make_session_factory
from propedge.db.store import PlayerResolver
from propedge.ingest.live import LivePoller, LiveStore
from propedge.odds.base import OddsProvider
from propedge.odds.kalshi import parse_market

FIX = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 10, 22, 4, 31, 30, tzinfo=timezone.utc)


def load(name):
    return json.loads((FIX / name).read_text())


@pytest.fixture
def Session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.sqlite'}")
    init_db(engine)
    return make_session_factory(engine)


def count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def test_game_state_is_stored_once_and_plays_are_not_duplicated(Session):
    state = parse_summary(load("espn_summary_live.json"), T0)
    store = LiveStore(Session(), stale_after_s=90)
    snap = store.store_game_state(state)
    s = store.session
    assert snap.game_id == "2026-10-21_GSW_LAL" and snap.feed_lag_s == 58 and not snap.is_stale
    assert count(s, m.Play) == 40
    assert all(pl.backlog for pl in s.scalars(select(m.Play)))   # first look at the game = backlog
    n_player_rows = count(s, m.LivePlayerSnapshot)
    assert n_player_rows == len(state.players)

    # Same payload a poll later: new game snapshot (to record lag), but no new plays or player rows.
    store.store_game_state(replace(state, fetched_at=T0 + timedelta(seconds=20)))
    assert count(s, m.LiveGameSnapshot) == 2
    assert count(s, m.Play) == 40
    assert count(s, m.LivePlayerSnapshot) == n_player_rows

    # Newest play 2 minutes old, but the game clock hasn't moved (timeout) -> not stale.
    snap3 = store.store_game_state(replace(state, fetched_at=T0 + timedelta(seconds=120)))
    assert snap3.clock_running is False and not snap3.is_stale

    # Clock moved but still no new plays -> our feed is behind -> stale.
    snap4 = store.store_game_state(replace(state, clock_s=state.clock_s - 30,
                                           fetched_at=T0 + timedelta(seconds=140)))
    assert snap4.clock_running is True and snap4.is_stale

    # Frozen feed: clock stuck too, but past the hard limit -> stale anyway.
    snap5 = store.store_game_state(replace(state, clock_s=state.clock_s - 30,
                                           fetched_at=T0 + timedelta(seconds=400)))
    assert snap5.clock_running is False and snap5.is_stale


def test_play_participants_are_mapped_to_our_player_ids(Session):
    state = parse_summary(load("espn_summary_live.json"), T0)
    store = LiveStore(Session(), stale_after_s=90)
    store.store_game_state(state)
    sub = store.session.scalars(select(m.Play).where(m.Play.play_type == "Substitution")).first()
    assert sub is not None and len(sub.player_ids) == 2
    names = [store.session.get(m.Player, pid).name for pid in sub.player_ids]
    assert sub.text.startswith(names[0])           # "<player in> enters the game for <player out>"


def test_espn_player_attaches_to_existing_nba_player(Session):
    s = Session()
    resolver = PlayerResolver(s)
    nba_pid = resolver.from_nba(201939, "Stephen Curry", "GSW")
    espn_pid = PlayerResolver(s).from_espn("3975", "Stephen Curry", "GSW", "PG")
    assert espn_pid == nba_pid
    assert s.get(m.Player, nba_pid).espn_athlete_id == "3975"


def test_odds_rows_only_written_when_quote_changes(Session):
    s = Session()
    PlayerResolver(s).from_nba(201939, "Stephen Curry", "GSW")
    s.commit()
    raw = load("kalshi_markets.json")["markets"]
    quotes = [parse_market(x, "pts", T0) for x in raw]
    store = LiveStore(s, stale_after_s=90)

    r1 = store.store_quotes(quotes)
    assert r1 == {"markets": 8, "changed": 8, "unchanged": 0}
    curry = s.scalars(select(m.PropMarket).where(m.PropMarket.player_name == "Stephen Curry")).all()
    assert curry and all(mk.player_id is not None for mk in curry)

    later = T0 + timedelta(seconds=30)
    r2 = store.store_quotes([replace(q, fetched_at=later) for q in quotes])
    assert r2["changed"] == 0 and count(s, m.OddsSnapshot) == 8
    assert all(row.last_seen_at.replace(tzinfo=timezone.utc) == later for row in s.scalars(select(m.OddsSnapshot)))

    moved = [replace(quotes[0], yes_ask=0.99, fetched_at=later + timedelta(seconds=30))] + \
            [replace(q, fetched_at=later + timedelta(seconds=30)) for q in quotes[1:]]
    r3 = store.store_quotes(moved)
    assert r3["changed"] == 1 and count(s, m.OddsSnapshot) == 9

    # A restarted poller picks up where the last one left off (no duplicate rows).
    r4 = LiveStore(s, stale_after_s=90).store_quotes(moved)
    assert r4["changed"] == 0


class FakeLive(LiveStatsProvider):
    name = "espn"

    def __init__(self, clock):
        self.clock = clock
        self.state = parse_summary(load("espn_summary_live.json"), T0)
        self.calls = {"scoreboard": 0, "game": 0}
        self.status = "in"

    def scoreboard(self, day=None):
        self.calls["scoreboard"] += 1
        return [ScheduledGame("401909092", date(2026, 10, 21), None, "GSW", "LAL", self.status)]

    def game_state(self, gid):
        self.calls["game"] += 1
        return replace(self.state, fetched_at=self.clock.now())


class FakeOdds(OddsProvider):
    name = "kalshi"

    def __init__(self, clock):
        self.clock, self.calls = clock, 0

    def fetch_quotes(self, stats):
        self.calls += 1
        return [parse_market(x, "pts", self.clock.now()) for x in load("kalshi_markets.json")["markets"]]


def test_poller_respects_poll_intervals_and_stops_when_idle(Session):
    cfg = load_config()
    clock = ReplayClock(T0)
    live, odds = FakeLive(clock), FakeOdds(clock)
    poller = LivePoller(Session, live, odds, cfg, ["pts"], now=clock.now, sleep=clock.sleep)
    poller.run(max_minutes=5)
    lf, oc = cfg["live_feed"], cfg["odds"]
    # 5 minutes of virtual time, plus the poll at t=0
    assert live.calls["scoreboard"] == 5 * 60 // lf["scoreboard_interval_s"] + 1
    assert live.calls["game"] == 5 * 60 // lf["poll_interval_s"] + 1
    assert odds.calls == 5 * 60 // oc["live_poll_interval_s"] + 1     # a game is live: fast odds polling

    live.status = "post"
    clock.sleep(lf["scoreboard_interval_s"])
    poller.run(stop_when_idle=True)
    assert poller.live_games == []


def test_replay_marks_everything_as_mock(Session, tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    for i in range(3):
        ts = T0 + timedelta(seconds=20 * i)
        stamp = ts.strftime("%Y%m%dT%H%M%SZ")
        with gzip.open(rec / f"espn_401909092_{stamp}.json.gz", "wt") as f:
            json.dump({"fetched_at": ts.isoformat(), "data": load("espn_summary_live.json")}, f)
        with gzip.open(rec / f"kalshi_KXNBAPTS_{stamp}.json.gz", "wt") as f:
            json.dump({"fetched_at": ts.isoformat(), "data": load("kalshi_markets.json")}, f)
    clock = ReplayClock(T0)
    cfg = load_config()
    poller = LivePoller(Session, ReplayLiveProvider(rec, clock),
                        ReplayOddsProvider(rec, clock, cfg["odds"]["kalshi"]["series"]),
                        cfg, ["pts"], now=clock.now, sleep=clock.sleep)
    poller.run(max_minutes=4)
    s = Session()
    assert count(s, m.LiveGameSnapshot) > 0
    assert all(x.is_mock for x in s.scalars(select(m.LiveGameSnapshot)))
    assert all(x.is_mock for x in s.scalars(select(m.Play)))
    assert all(x.is_mock and x.provider == "replay" for x in s.scalars(select(m.PropMarket)))
    assert all(x.is_mock for x in s.scalars(select(m.OddsSnapshot)))


def test_recorder_output_is_replayable(Session, tmp_path):
    from propedge.data.recorder import RawRecorder

    rec = RawRecorder(tmp_path / "recordings")
    rec("espn", "401909092", T0, load("espn_summary_live.json"))
    rec("kalshi", "KXNBAPTS", T0, load("kalshi_markets.json"))
    folder = tmp_path / "recordings" / "2026-10-22"   # 04:31 UTC = 00:31 Eastern on Oct 22
    clock = ReplayClock(T0)
    live = ReplayLiveProvider(folder, clock)
    assert live.game_state("401909092").away == "GSW"
    odds = ReplayOddsProvider(folder, clock, {"pts": "KXNBAPTS"})
    assert len(odds.fetch_quotes(["pts"])) == 8


def test_poller_runs_through_pregame_and_stops_when_all_final(Session):
    cfg = load_config()
    clock = ReplayClock(T0)
    live, odds = FakeLive(clock), FakeOdds(clock)
    live.status = "pre"
    poller = LivePoller(Session, live, odds, cfg, ["pts"], now=clock.now, sleep=clock.sleep)
    calls = []
    poller.jobs.append((timedelta(minutes=10), lambda now: calls.append(now)))
    poller.run(max_minutes=3, stop_when_all_final=True)      # pre-game: keeps polling odds
    assert odds.calls > 1 and live.calls["game"] == 0 and len(calls) == 1
    live.status = "post"
    clock.sleep(cfg["live_feed"]["scoreboard_interval_s"])
    poller.run(max_minutes=60, stop_when_all_final=True)
    assert poller.all_final
    assert clock.now() < T0 + timedelta(minutes=10)           # stopped promptly, not after 60 min


def test_closing_quote_is_the_last_quote_before_tip(Session):
    from propedge.ingest.closing import capture_from_snapshots, tip_time

    s = Session()
    state = parse_summary(load("espn_summary_live.json"), T0)
    store = LiveStore(s, stale_after_s=90)
    raw = load("kalshi_markets.json")["markets"]
    tip = T0 - timedelta(minutes=5)
    before = [parse_market(x, "pts", tip - timedelta(minutes=10)) for x in raw]
    store.store_quotes(before)
    after = [replace(q, yes_ask=0.99, fetched_at=tip + timedelta(minutes=1)) for q in before]   # in-game move
    store.store_quotes(after)
    store.store_game_state(state)
    assert tip_time(s, "2026-10-21_GSW_LAL") is not None
    n = capture_from_snapshots(s, "2026-10-21_GSW_LAL", tip)
    assert n == len(before)
    rows = {r.ticker: r for r in s.scalars(select(m.ClosingQuote))}
    q0 = before[0]
    assert rows[q0.ticker].yes_ask == q0.yes_ask          # the pre-tip quote, not the in-game 0.99


def test_neutral_site_game_listed_the_other_way_round_reuses_the_row(Session):
    from propedge.db.store import ensure_game

    s = Session()
    gid = ensure_game(s, date(2025, 12, 13), "OKC", "SAS", nba_game_id="0022501234")
    s.commit()
    gid2 = ensure_game(s, date(2025, 12, 13), "SAS", "OKC", espn_event_id="401809838", away_score=111,
                       home_score=109, home_spread=-10.5)
    s.commit()
    g = s.get(m.Game, gid)
    assert gid2 == gid and g.espn_event_id == "401809838"
    assert (g.away_score, g.home_score, g.home_spread) == (109, 111, 10.5)


def test_failed_scoreboard_fetch_never_means_all_games_final(Session):
    cfg = load_config()
    clock = ReplayClock(T0)
    live, odds = FakeLive(clock), FakeOdds(clock)
    live.status = "pre"
    real = live.scoreboard
    live.scoreboard = lambda day=None: None          # network down: every fetch fails
    poller = LivePoller(Session, live, odds, cfg, ["pts"], now=clock.now, sleep=clock.sleep)
    poller.run(max_minutes=10, stop_when_all_final=True)
    assert not poller.all_final and clock.now() >= T0 + timedelta(minutes=10)   # kept going
    live.scoreboard = real                           # network back
    poller.run(max_minutes=2, stop_when_all_final=True)
    assert not poller.all_final                      # games still "pre"
