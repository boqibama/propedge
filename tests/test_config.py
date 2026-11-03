from propedge.config import load_config


def test_config_has_every_source_setting():
    cfg = load_config()
    required = {"min_interval_s", "timeout_s", "max_retries", "backoff_s", "cache_ttl_s"}
    for name, settings in cfg["sources"].items():
        assert required <= settings.keys(), name


def test_thresholds_are_sane():
    cfg = load_config()
    assert 0 < cfg["edge"]["min_edge"] < 0.5
    assert cfg["edge"]["no_flag"]["after_price_move_s"] >= cfg["live_feed"]["stale_after_s"]
    assert 0 < cfg["odds"]["kalshi"]["fees"]["taker_coef"] < 0.2
    assert 0 < cfg["bankroll"]["kelly_fraction"] <= 1
    assert cfg["model"]["n_sims"] >= 1000
