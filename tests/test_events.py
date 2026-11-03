"""Event detection in the live engine (4th fouls, blowout onset) and the team constraints."""

from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from propedge.config import load_config
from propedge.db.session import init_db, make_engine, make_session_factory
from propedge.edge.live import GameContext, LiveEngine

CFG = load_config()
NOW = datetime(2026, 10, 24, 23, 0, tzinfo=timezone.utc)


@dataclass
class Line:
    pf: int
    team: str


@dataclass
class State:
    home_score: int
    away_score: int
    period: int


@pytest.fixture
def engine(tmp_path):
    e = make_engine(f"sqlite:///{tmp_path / 't.sqlite'}")
    init_db(e)
    return LiveEngine(make_session_factory(e)(), CFG, log_flags=False, record=False)


def ctx():
    pre = pd.DataFrame({"min_mu": [34.0, 30.0, 12.0, 33.0]}, index=[1, 2, 3, 4])
    return GameContext("2026-10-24_AAA_HHH", "HHH", pre)


def box(pf1=3):
    return {1: Line(pf1, "HHH"), 2: Line(0, "HHH"), 3: Line(1, "HHH"), 4: Line(2, "AAA")}


def test_fourth_foul_triggers_player_and_teammates_only(engine):
    g = ctx()
    assert engine.detect_events(g.game_id, State(50, 48, 3), box(3), g, 30.0, NOW) == {}   # first look: record
    scope = engine.detect_events(g.game_id, State(52, 48, 3), box(4), g, 30.5, NOW)
    assert scope[1]["type"] == "foul4" and scope[1]["role"] == "fouler"
    assert scope[2]["role"] == scope[3]["role"] == "teammate"
    assert 4 not in scope                                                      # other team untouched
    assert engine.detect_events(g.game_id, State(54, 48, 3), box(5), g, 31.0, NOW) == {}   # 5th foul isn't foul4


def test_existing_fouls_at_first_look_are_not_events(engine):
    g = ctx()
    assert engine.detect_events(g.game_id, State(50, 48, 3), box(4), g, 30.0, NOW) == {}


def test_blowout_fires_once_when_lead_first_reaches_threshold_in_second_half(engine):
    g = ctx()
    lead = CFG["repricing"]["blowout_lead"]
    engine.detect_events(g.game_id, State(60, 60 - lead + 2, 3), box(0), g, 30.0, NOW)
    scope = engine.detect_events(g.game_id, State(62, 62 - lead, 3), box(0), g, 31.0, NOW)
    assert {v["type"] for v in scope.values()} == {"blowout"}
    assert scope[1]["role"] == "starter" and scope[3]["role"] == "bench" and scope[4]["role"] == "starter"
    assert engine.detect_events(g.game_id, State(64, 64 - lead - 3, 3), box(0), g, 32.0, NOW) == {}   # only once


def test_no_blowout_event_in_first_half_or_when_already_a_blowout_at_first_look(engine):
    g = ctx()
    lead = CFG["repricing"]["blowout_lead"]
    engine.detect_events(g.game_id, State(40, 40 - lead + 2, 2), box(0), g, 20.0, NOW)
    assert engine.detect_events(g.game_id, State(44, 44 - lead, 2), box(0), g, 21.0, NOW) == {}     # first half
    g2 = GameContext("2026-10-24_BBB_CCC", "CCC", g.pre)
    assert engine.detect_events(g2.game_id, State(80, 80 - lead - 5, 3), box(0), g2, 33.0, NOW) == {}  # restart


def test_team_constraints_trim_minutes_and_points_but_never_raise():
    from propedge.models.points_model import project
    p = dict(CFG["model"]["points"])
    n = 13
    f = pd.DataFrame({
        "ewm_min": np.full(n, 26.0), "ewm_min_sd": np.full(n, 5.0), "days_rest": np.full(n, 2),
        "exp_margin": np.zeros(n), "missing_share": np.zeros(n), "career_min": np.full(n, 3000.0),
        "rate_obs": np.full(n, 0.6), "opp_def_factor": np.ones(n), "pace_factor": np.ones(n),
        "is_home": np.ones(n, bool), "rookie": np.zeros(n, bool), "new_team": np.zeros(n, bool),
        "returning": np.zeros(n, bool), "history_unknown": np.zeros(n, bool), "team_new_share": np.zeros(n),
        "games_before": np.full(n, 20), "team": "HHH", "game_id": "g1", "team_exp_pts": np.full(n, 110.0)})
    proj = project(f, p)
    assert proj.min_mu.sum() <= p["team_minutes_cap"] + 1e-6           # 13 × 26.5 = 345 -> trimmed to 240
    assert (proj.rate * proj.min_mu).sum() <= 110.0 * p["team_points_share"] + 1e-6
    small = project(f.iloc[:5].copy(), p)                                # 5 players: well under both caps…
    free = project(f.iloc[:5].copy(), dict(p, team_constraints=False))   # …so nothing changes
    assert np.allclose(small.min_mu, free.min_mu) and np.allclose(small.rate, free.rate)
