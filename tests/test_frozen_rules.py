"""The frozen test (DECISIONS #62) must not drift. Changing any value here needs the user's explicit OK
and a new DECISIONS entry — this test exists so that can't happen by accident."""

from propedge.config import load_config

FROZEN = {
    "judge_from": "2026-10-26",
    "entry_delay_s": 60,
    "max_quote_age_s": 30,
    "blowout_lead": 20,
    "blowout_from_min": 24,
    "foul_n": 4,
    "rung_max_spread": 0.06,
    "rule_a": {"event": "foul4", "max_period": 3, "players": "fouler", "side": "no", "rung": "closest_to_50",
               "hold_s": 300},
    "rule_b": {"event": "blowout", "players": "starter", "side": "no", "rung": "closest_to_50", "hold_s": 120},
    "decision_events": 50,
}


def test_frozen_rules_unchanged():
    assert load_config()["frozen_rules"] == FROZEN


def test_frozen_rules_fee_assumptions():
    k = load_config()["odds"]["kalshi"]
    assert k["order_contracts"] == 10
    assert k["fees"]["taker_coef"] == 0.07 and k["fees"]["fee_type"]["pts"] == "quadratic"
