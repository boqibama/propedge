"""Points model: distribution math and — most important — no look-ahead in the features."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from propedge.config import load_config
from propedge.models.points_dist import (PointsDistribution, expected_minutes, minutes_weights, points_pmf,
                                         prob_at_least)
from propedge.models.points_features import build_features
from propedge.models.points_model import blowout_probability, project

PARAMS = load_config()["model"]["points"]


# ----------------------------------------------------------------------------- distribution

def test_minutes_weights_are_a_distribution_on_1_to_48():
    w = minutes_weights(np.array([30.0, 2.0, 47.0, 80.0]), np.array([5.0, 3.0, 4.0, 1.0]))
    assert w.shape == (4, 48)
    assert np.allclose(w.sum(axis=1), 1)
    assert np.isclose(expected_minutes(np.array([30.0]), np.array([0.5]))[0], 30, atol=0.01)
    assert w[3, -1] == 1.0                     # far beyond 48: everything at 48


def test_pmf_sums_to_one_and_mean_matches_rate_times_minutes():
    rate, mu, sd = np.array([0.5, 0.8]), np.array([30.0, 20.0]), np.array([4.0, 6.0])
    pmf = points_pmf(rate, mu, sd, phi=2.5, rate_cv=np.array([0.05, 0.1]))
    assert np.allclose(pmf.sum(axis=1), 1)
    means = pmf @ np.arange(pmf.shape[1])
    assert np.allclose(means, rate * expected_minutes(mu, sd), rtol=0.01)


def test_more_dispersion_means_wider_distribution():
    args = (np.array([0.6]), np.array([30.0]), np.array([4.0]))
    narrow = PointsDistribution(points_pmf(*args, phi=1.5, rate_cv=np.array([0.0]))[0])
    wide = PointsDistribution(points_pmf(*args, phi=3.5, rate_cv=np.array([0.2]))[0])
    assert wide.sd > narrow.sd
    assert abs(wide.mean - narrow.mean) < 0.2


def test_p_over_is_monotone_and_matches_ladder_convention():
    d = PointsDistribution(points_pmf(np.array([0.6]), np.array([32.0]), np.array([4.0]), 2.5, np.array([0.05]))[0])
    probs = [d.p_over(x + 0.5) for x in range(0, 50)]
    assert all(a >= b for a, b in zip(probs, probs[1:]))
    # Kalshi "25+" == over 24.5 == P(points >= 25)
    assert d.p_over(24.5) == pytest.approx(prob_at_least(d.pmf[None, :], 25)[0])


def test_blowout_probability_grows_with_expected_margin():
    p = dict(PARAMS)
    probs = blowout_probability(np.array([0.0, 5.0, 10.0, 20.0, -20.0]), p)
    assert probs[0] < probs[1] < probs[2] < probs[3]
    assert probs[3] == pytest.approx(probs[4])


# ----------------------------------------------------------------------------- synthetic league

def synthetic_league(n_games=30, seed=1):
    """Two teams, 8 players each, playing each other every other day across two seasons."""
    rng = np.random.default_rng(seed)
    players, teams = [], []
    day0 = date(2024, 10, 22)
    for season, offset in (("2024-25", 0), ("2025-26", 365)):
        for g in range(n_games):
            d = day0 + timedelta(days=offset + 2 * g)
            gid = f"{d}_AAA_BBB"
            for team, opp, home in (("AAA", "BBB", False), ("BBB", "AAA", True)):
                tot = 0
                for k in range(8):
                    pid = (0 if team == "AAA" else 100) + k
                    mins = float(np.clip(rng.normal(34 - 3 * k, 4), 1, 46))
                    pts = int(rng.poisson(0.5 * mins))
                    tot += pts
                    players.append(dict(player_id=pid, game_id=gid, season=season, game_date=d, team=team,
                                        opponent=opp, is_home=home, minutes=mins, pts=pts))
                teams.append(dict(game_id=gid, season=season, game_date=d, team=team, opponent=opp, is_home=home,
                                  minutes=240.0, pts=tot, fga=85.0, fta=22.0, oreb=10.0, tov=14.0))
    games = pd.DataFrame({"game_id": sorted({t["game_id"] for t in teams})}).assign(home_spread=np.nan)
    return pd.DataFrame(players), pd.DataFrame(teams), games


FEATURES = ["ewm_min", "ewm_min_sd", "rate_obs", "career_min", "baseline_season_avg", "baseline_last10",
            "exp_margin", "pace_factor", "opp_def_factor", "missing_share", "team_new_share"]


def test_features_never_use_the_current_or_future_games():
    players, teams, games = synthetic_league()
    f1 = build_features(players, teams, games, PARAMS).sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Change everything from one date onward (huge scores, one player missing): features up to
    # and including that date must not move.
    cut = sorted(players["game_date"].unique())[40]
    p2, t2 = players.copy(), teams.copy()
    later = p2["game_date"] >= cut
    p2.loc[later, "pts"] = 99
    p2.loc[later, "minutes"] = 45.0
    p2 = p2[~(later & (p2["player_id"] == 3))]
    t2.loc[t2["game_date"] >= cut, "pts"] = 300
    f2 = build_features(p2, t2, games, PARAMS).sort_values(["player_id", "game_date"]).reset_index(drop=True)

    a = f1[f1["game_date"] <= cut].set_index(["player_id", "game_id"])[FEATURES]
    b = f2[f2["game_date"] <= cut].set_index(["player_id", "game_id"])[FEATURES]
    common = a.index.intersection(b.index)
    cols_ok = [c for c in FEATURES if c != "missing_share"]   # missing_share on `cut` legitimately sees who is absent that day
    pd.testing.assert_frame_equal(a.loc[common, cols_ok], b.loc[common, cols_ok])


def test_projecting_an_unplayed_game_equals_the_pre_game_view_of_a_played_one():
    players, teams, games = synthetic_league()
    last_day = players["game_date"].max()
    f_played = build_features(players, teams, games, PARAMS)
    target = players[players["game_date"] == last_day].assign(minutes=np.nan, pts=np.nan)
    hist = players[players["game_date"] < last_day]
    tteams = teams.copy()
    tteams.loc[tteams["game_date"] == last_day, ["minutes", "pts", "fga", "fta", "oreb", "tov"]] = np.nan
    f_target = build_features(pd.concat([hist, target]), tteams, games, PARAMS)
    cols = ["ewm_min", "rate_obs", "career_min", "baseline_season_avg", "exp_margin", "pace_factor"]
    a = f_played[f_played["game_date"] == last_day].set_index("player_id")[cols].sort_index()
    b = f_target[f_target["game_date"] == last_day].set_index("player_id")[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_projection_is_sane_on_synthetic_data():
    players, teams, games = synthetic_league()
    f = build_features(players, teams, games, PARAMS)
    f = f[(f["season"] == "2025-26") & f["ewm_min"].notna()]
    proj = project(f, PARAMS)
    err = np.abs(proj.mean_points - f["pts"].to_numpy()).mean()
    naive = np.abs(f["baseline_last10"].to_numpy() - f["pts"].to_numpy()).mean()
    assert err < naive * 1.1                       # in the same league as a simple average
    assert not proj.low_confidence.all()


def test_new_team_and_rookie_are_low_confidence():
    players, teams, games = synthetic_league()
    # Player 7 switches from AAA to BBB for season two.
    moved = (players["player_id"] == 7) & (players["season"] == "2025-26")
    players.loc[moved, ["team", "opponent", "is_home"]] = ["BBB", "AAA", True]
    # Player 200 debuts in season two.
    debut = players[(players["player_id"] == 106) & (players["season"] == "2025-26")].assign(player_id=200)
    players = pd.concat([players, debut], ignore_index=True)
    f = build_features(players, teams, games, PARAMS)
    f = f[f["ewm_min"].notna()].reset_index(drop=True)
    proj = project(f, PARAMS)
    reasons = pd.Series(proj.low_reasons, index=f.index)
    first_new = f[(f["player_id"] == 7) & (f["team"] == "BBB")].sort_values("game_date").index[0]
    assert "new_team" in reasons[first_new]
    rookie_rows = f[f["player_id"] == 200].sort_values("game_date").index
    assert "rookie" in reasons[rookie_rows[0]]
    assert "rookie" not in reasons[rookie_rows[-1]]    # after enough games


def test_seeded_season_ewm_restarts_each_season_from_last_seasons_average():
    from propedge.models.points_features import seeded_season_ewm

    vals = np.array([36, 36, 36, 18, 34, 34, np.nan])          # last game of season 1 was a rest night
    season = np.array(["s1", "s1", "s1", "s1", "s2", "s2", "s2"])
    seed = np.array([np.nan] * 4 + [31.5] * 3)                # season-1 mpg
    out = seeded_season_ewm(vals, season, seed, halflife=3, seed_games=5)
    assert np.isnan(out[0])                                   # nothing known before the first game
    assert out[3] == pytest.approx(36)                        # uses only earlier games
    assert out[4] == pytest.approx(31.5)                      # new season starts from last season's mpg, not the 18
    assert 31.5 < out[5] < 34 and out[6] > out[5]             # then moves toward this season's games
    no_seed = seeded_season_ewm(vals, season, seed, halflife=3, seed_games=0)
    assert np.isnan(no_seed[4])                               # seed_games=0 -> no seed
