"""Bias correction, flag statuses, scoring rules, blend weight and CLV."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from propedge.edge.pregame import assign_statuses
from propedge.edge.scoreboard import fit_blend_weight
from propedge.models.bias import bias_factor, rolling_bias_factors
from propedge.models.scoring import brier, log_loss


# ----------------------------------------------------------------------------- bias

def test_bias_factor_direction_shrinkage_and_cap():
    proj = np.full(100, 10.0)
    assert bias_factor(proj, np.full(100, 9.0), prior_games=0, cap=0.5) == pytest.approx(0.9)   # model 10% high
    assert bias_factor(proj, np.full(100, 9.0), prior_games=100, cap=0.5) == pytest.approx(0.95)  # half the data
    assert bias_factor(proj, np.full(100, 5.0), prior_games=0, cap=0.02) == pytest.approx(0.98)  # capped
    assert bias_factor([], [], 100, 0.02) == 1.0


def test_rolling_bias_uses_only_earlier_days_of_the_same_season():
    d = pd.to_datetime(["2025-11-01"] * 3 + ["2025-11-02"] * 3 + ["2026-11-01"] * 3)
    season = pd.Series(["2025-26"] * 6 + ["2026-27"] * 3)
    proj = np.full(9, 10.0)
    actual = np.array([8, 8, 8, 30, 30, 30, 10, 10, 10], float)   # day 2 explodes: must not affect day 1 or 2
    f = rolling_bias_factors(season, pd.Series(d), proj, actual, prior_games=0, cap=0.5, window_days=400)
    assert np.allclose(f[:3], 1.0)              # first day: no history
    assert np.allclose(f[3:6], 0.8)             # second day sees only day 1 (20% high)
    assert np.allclose(f[6:], 1.0)              # new season starts fresh


# ----------------------------------------------------------------------------- statuses

def test_statuses_one_flag_per_player_best_edge():
    c = pd.DataFrame([
        dict(player_id=1, edge=0.20, passes=True, reasons=[]),
        dict(player_id=1, edge=0.15, passes=True, reasons=[]),
        dict(player_id=2, edge=0.30, passes=True, reasons=[{"code": "low_confidence"}]),
        dict(player_id=3, edge=0.05, passes=False, reasons=[]),
    ])
    assert assign_statuses(c, one_per_player=True) == ["flag", "not_best", "suppressed", "below_threshold"]
    assert assign_statuses(c, one_per_player=False) == ["flag", "flag", "suppressed", "below_threshold"]


# ----------------------------------------------------------------------------- scoring

def test_brier_and_log_loss():
    y = np.array([1.0, 0.0])
    assert brier(np.array([1.0, 0.0]), y) == 0
    assert brier(np.array([0.5, 0.5]), y) == pytest.approx(0.25)
    assert log_loss(np.array([0.5, 0.5]), y) == pytest.approx(np.log(2))
    assert np.isfinite(log_loss(np.array([0.0, 1.0]), y))       # clipped, not infinite


def test_blend_weight_picks_the_better_forecaster():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < truth).astype(float)
    good = truth
    noisy = np.clip(truth + rng.normal(0, 0.3, 4000), 0.01, 0.99)
    assert fit_blend_weight(good, noisy, y) >= 0.8
    assert fit_blend_weight(noisy, good, y) <= 0.2


# ----------------------------------------------------------------------------- CLV

def test_clv_sign(tmp_path):
    from propedge.db import models as m
    from propedge.db.session import init_db, make_engine, make_session_factory
    from propedge.edge.scoreboard import flag_clv

    engine = make_engine(f"sqlite:///{tmp_path / 't.sqlite'}")
    init_db(engine)
    s = make_session_factory(engine)()
    tip = datetime(2026, 10, 22, 23, 0, tzinfo=timezone.utc)
    s.add(m.Game(id="2026-10-22_CLE_PHI", season="2026-27", game_date=date(2026, 10, 22), away="CLE", home="PHI"))
    s.add(m.Player(id=1, name="A", name_key="a"))
    s.flush()
    for t in ("T1", "T2"):
        s.add(m.PropMarket(ticker=t, provider="kalshi", event_ticker="E", game_id="2026-10-22_CLE_PHI", player_id=1,
                           player_name="A", stat="pts", line=19.5))
        s.add(m.ClosingQuote(ticker=t, game_id="2026-10-22_CLE_PHI", tip_time=tip, mid=0.50, source="poll"))
    s.flush()                                    # parents before flags (foreign keys)
    common = dict(kind="pregame", reasons=[], game_id="2026-10-22_CLE_PHI", player_id=1, stat="pts", line=19.5,
                  model_prob=0.6, fee_per_contract=0.02, edge=0.1, ev_per_dollar=0.2, confidence="normal", inputs={})
    s.add(m.Flag(created_at=tip - timedelta(hours=2), status="flag", ticker="T1", side="yes", ask=0.40, cost=0.42, **common))
    s.add(m.Flag(created_at=tip - timedelta(hours=2), status="suppressed", ticker="T2", side="no", ask=0.40, cost=0.42,
                 **common))
    s.commit()
    df = flag_clv(s).set_index("ticker")
    assert df.loc["T1", "clv"] == pytest.approx(0.10)        # bought YES at 0.40, closed at 0.50
    assert df.loc["T2", "clv"] == pytest.approx(0.10)        # bought NO at 0.40, NO closed at 1 − 0.50
    assert df.loc["T1", "clv_after_fee"] == pytest.approx(0.08)
