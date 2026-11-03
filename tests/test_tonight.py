from datetime import date, datetime, timedelta, timezone

from propedge.data.live_base import ScheduledGame
from propedge.data.replay import ReplayClock
from propedge.ingest.tonight import plan_start, wait_until

NOW = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
LEAD = timedelta(minutes=90)


def game(status, tip_hours_from_now):
    return ScheduledGame("1", date(2026, 10, 22), NOW + timedelta(hours=tip_hours_from_now), "GSW", "LAL", status)


def test_starts_lead_minutes_before_first_tip():
    games = [game("pre", 5), game("pre", 3), game("pre", 6)]
    assert plan_start(games, NOW, LEAD) == NOW + timedelta(hours=3) - LEAD


def test_starts_now_if_a_game_is_live_or_tip_is_close():
    assert plan_start([game("in", -1), game("pre", 2)], NOW, LEAD) == NOW
    assert plan_start([game("pre", 1)], NOW, LEAD) == NOW


def test_nothing_to_do_when_all_final_or_no_games():
    assert plan_start([game("post", -3)], NOW, LEAD) is None
    assert plan_start([], NOW, LEAD) is None


def test_finished_games_are_ignored_when_planning():
    assert plan_start([game("post", -3), game("pre", 4)], NOW, LEAD) == NOW + timedelta(hours=4) - LEAD


def test_wait_until_sleeps_to_target():
    clock = ReplayClock(NOW)
    wait_until(NOW + timedelta(minutes=30), clock.now, clock.sleep, timedelta(minutes=15))
    assert clock.now() == NOW + timedelta(minutes=30)


def _backfill_env(monkeypatch, tmp_path, reachable: bool):
    import argparse
    from propedge import cli
    from propedge.config import load_config
    from propedge.data import nba_stats
    from propedge.ingest import history

    calls = []
    monkeypatch.setattr(nba_stats.NbaStatsClient, "reachable", lambda self, timeout_s=15: reachable)
    monkeypatch.setattr(cli, "cmd_backfill", lambda args, cfg: calls.append("full"))
    monkeypatch.setattr(history, "ingest_schedule", lambda *a, **k: calls.append("schedule") or 0)
    monkeypatch.setattr(history, "ingest_historical_spreads", lambda *a, **k: calls.append("spreads") or 0)
    args = argparse.Namespace(db=f"sqlite:///{tmp_path / 't.sqlite'}")
    return cli, args, load_config(), calls


def test_tonight_backfill_full_when_nba_reachable(monkeypatch, tmp_path):
    cli, args, cfg, calls = _backfill_env(monkeypatch, tmp_path, reachable=True)
    cli._tonight_backfill(args, cfg)
    assert calls == ["full"]


def test_tonight_backfill_espn_only_when_nba_blocked(monkeypatch, tmp_path):
    cli, args, cfg, calls = _backfill_env(monkeypatch, tmp_path, reachable=False)
    cli._tonight_backfill(args, cfg)
    assert calls[0] == "schedule" and "full" not in calls


def test_tonight_backfill_failure_never_stops_polling(monkeypatch, tmp_path):
    cli, args, cfg, calls = _backfill_env(monkeypatch, tmp_path, reachable=True)

    def boom(args, cfg):
        raise RuntimeError("stats.nba.com hung up")
    monkeypatch.setattr(cli, "cmd_backfill", boom)
    cli._tonight_backfill(args, cfg)          # logged, not raised
