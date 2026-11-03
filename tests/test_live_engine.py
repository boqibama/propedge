"""Game-state reconstruction and the live points engine."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from propedge.config import load_config
from propedge.data.live_base import LivePlay
from propedge.models.game_state import elapsed_minutes, is_personal_foul, replay_states
from propedge.models.live_points import LiveInputs, blowout_meter, simulate

CFG = load_config()
P = {**CFG["model"]["points"], **CFG["live"]}
T0 = datetime(2026, 10, 22, 23, 0, tzinfo=timezone.utc)


def play(seq, period, clock, ptype, ids, scoring=False, value=0, team="AAA"):
    return LivePlay(str(seq), seq, period, clock, T0 + timedelta(seconds=seq * 20), ptype, "", team, scoring,
                    value, 0, 0, tuple(ids))


def test_elapsed_minutes():
    assert elapsed_minutes(1, 720) == 0
    assert elapsed_minutes(2, 360) == 18
    assert elapsed_minutes(4, 0) == 48
    assert elapsed_minutes(5, 0) == 53


def test_personal_foul_detection():
    assert is_personal_foul("Shooting Foul") and is_personal_foul("Loose Ball Foul")
    assert is_personal_foul("Offensive Foul") and is_personal_foul("Flagrant Foul Type 1")
    assert not is_personal_foul("Offensive Foul Turnover") and not is_personal_foul("Technical Foul")


def test_replay_tracks_points_fouls_minutes_and_substitutions():
    team_of = {f"a{i}": "AAA" for i in range(7)}
    starters = {"AAA": {f"a{i}" for i in range(5)}}
    plays = [
        play(1, 1, 700, "Jump Shot", ["a0"], scoring=True, value=2),
        play(2, 1, 600, "Shooting Foul", ["a1"]),
        play(3, 1, 360, "Substitution", ["a5", "a0"]),          # a5 in for a0 at 6:00 of Q1
        play(4, 1, 100, "Free Throw - 1 of 1", ["a5"], scoring=True, value=1),
        play(5, 2, 700, "Jump Shot", ["a0"], scoring=True, value=3),   # a0 back for Q2 (first event isn't "in")
    ]
    st = replay_states(plays, starters, team_of, [6.0, 12.0, 13.0])
    s6, s12, s13 = st
    assert s6.players["a0"].pts == 2 and s6.players["a1"].fouls == 1
    assert s6.players["a0"].minutes == pytest.approx(6.0)
    assert s6.players["a0"].on_court is False and s6.players["a5"].on_court is True   # sub at 6:00 applied
    assert s12.players["a0"].minutes == pytest.approx(6.0)       # sat 6:00–12:00 of Q1
    assert s12.players["a5"].minutes == pytest.approx(6.0)
    assert s12.players["a5"].pts == 1
    assert s13.players["a0"].on_court and s13.players["a0"].pts == 5


def base_inputs(n=1, **kw):
    x = dict(pts=np.full(n, 10.0), fouls=np.zeros(n), minutes=np.full(n, 16.0), on_court=np.ones(n, bool),
             off_for=np.zeros(n), margin=np.zeros(n), rate_pre=np.full(n, 0.6), min_pre=np.full(n, 34.0),
             min_sd=np.full(n, 4.0), pf_rate=np.full(n, 0.07), elapsed=24.0, period=2, period_left=0.0)
    x.update(kw)
    return LiveInputs(**x)


def mean_final(x, seed=0):
    return simulate(x, P, 20000, np.random.default_rng(seed)).mean(axis=1)


def test_halftime_projection_is_sensible():
    m = mean_final(base_inputs())[0]
    assert 18 < m < 23          # 10 so far + ~0.6 × ~17 more minutes


def test_points_never_go_down_and_decided_lines_are_certain():
    final = simulate(base_inputs(), P, 5000, np.random.default_rng(1))
    assert (final >= 10).all()
    assert (final > 9.5).mean() == 1.0


def test_blowout_cuts_starters_minutes():
    calm = mean_final(base_inputs(margin=np.array([2.0]), elapsed=40.0, period=4, period_left=8.0, minutes=np.array([28.0])))[0]
    rout = mean_final(base_inputs(margin=np.array([28.0]), elapsed=40.0, period=4, period_left=8.0, minutes=np.array([28.0])))[0]
    assert rout < calm - 1.5
    assert blowout_meter(np.array([28.0]), 8.0, P)[0] > 0.9 > blowout_meter(np.array([2.0]), 8.0, P)[0]


def test_foul_trouble_and_foul_out():
    early = dict(elapsed=6.0, period=1, period_left=6.0, minutes=np.array([6.0]), pts=np.array([4.0]))
    clean = mean_final(base_inputs(**early))[0]
    two_fouls = mean_final(base_inputs(fouls=np.array([2.0]), **early))[0]
    assert two_fouls < clean
    fouled_out = mean_final(base_inputs(fouls=np.array([6.0])))[0]
    assert fouled_out == pytest.approx(10.0)


def test_long_absence_of_a_starter_means_probably_gone():
    here = mean_final(base_inputs())[0]
    gone = mean_final(base_inputs(on_court=np.array([False]), off_for=np.array([20.0])))[0]
    assert gone < here - 3


def test_hot_start_is_shrunk_toward_pregame_rate():
    hot = base_inputs(pts=np.array([20.0]), minutes=np.array([12.0]), elapsed=12.0, period=1)
    parts = simulate(hot, P, 2000, np.random.default_rng(2), return_parts=True)[1]
    assert 0.6 < parts["rate"][0] < 0.8          # 1.67/min tonight, 0.6 pre-game -> mostly pre-game
