"""Command line: `python -m propedge <command>`.

    init-db     create the database tables
    backfill    historical game logs, team stats, matchups, schedule
    poll        live poller (ESPN game state + Kalshi quotes) — run during games
    tonight     the whole night: backfill, wait for tip-off, poll + record until all games end
    replay      run the poller over a recorded night (into a separate mock database)
    status      what's in the database, feed lag, unmatched players
"""

from __future__ import annotations

import argparse
import logging
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select

from propedge.config import ROOT, load_config
from propedge.db import models as m
from propedge.db.session import init_db, make_engine, make_session_factory
from propedge.db.store import utc
from propedge.logging_setup import setup_logging

log = logging.getLogger("propedge")

THIN_SLICE_STATS = ["pts"]   # points only until the thin slice works end to end
REPLAY_DB = "sqlite:///data/replay.sqlite"


def _session_factory(db_url: str | None):
    engine = make_engine(db_url)
    init_db(engine)
    return make_session_factory(engine)


def cmd_init_db(args, cfg) -> None:
    _session_factory(args.db)
    log.info("database ready: %s", args.db or cfg["paths"]["database_url"])


def cmd_backfill(args, cfg) -> None:
    from propedge.data.espn import EspnLiveProvider
    from propedge.data.nba_stats import NbaStatsClient
    from propedge.ingest.history import backfill

    hist = cfg["history"]
    Session = _session_factory(args.db)
    with Session() as session:
        counts = backfill(session, NbaStatsClient(cfg), EspnLiveProvider(),
                          seasons=args.seasons or hist["seasons"],
                          matchup_seasons=[] if args.skip_matchups else hist["backfill_matchups_for"],
                          schedule_days_back=hist["schedule_days_back"],
                          schedule_days_ahead=hist["schedule_days_ahead"],
                          spread_seasons=[] if args.skip_spreads else hist["spreads_for"], refresh=args.refresh)
    for k, v in counts.items():
        log.info("backfill %-24s %d rows", k, v)


def cmd_poll(args, cfg) -> None:
    from propedge.data.espn import EspnLiveProvider
    from propedge.ingest.live import LivePoller
    from propedge.odds.kalshi import KalshiOddsProvider

    recorder = None
    if args.record:
        from propedge.data.recorder import RawRecorder
        recorder = RawRecorder(ROOT / cfg["paths"]["data_dir"] / "recordings", cfg["recording"]["min_interval_s"])
        log.info("recording raw payloads under data/recordings/ (for `propedge replay`)")
    poller = LivePoller(_session_factory(args.db), EspnLiveProvider(on_raw=recorder),
                        None if args.no_odds else KalshiOddsProvider(config=cfg, on_raw=recorder),
                        cfg, THIN_SLICE_STATS)
    if not args.no_engine:
        from propedge.edge.live import LiveEngine
        poller.engine = LiveEngine(poller.store.session, cfg)  # paper predictions + flags during games
    try:
        poller.run(max_minutes=args.minutes, stop_when_idle=args.until_idle)
    except KeyboardInterrupt:
        log.info("poller stopped (Ctrl-C)")


def _tonight_backfill(args, cfg) -> None:
    """Backfill before polling, but never at the cost of polling: if stats.nba.com is unreachable (typical
    on a cloud server) only the ESPN parts run (schedule, spreads), and any failure is logged and skipped.
    Without NBA logs, settlement falls back to ESPN's final box scores and the model's history goes stale;
    the frozen test (DECISIONS #62) doesn't use either."""
    from propedge.data.espn import EspnLiveProvider
    from propedge.data.nba_stats import NbaStatsClient
    from propedge.ids import EASTERN
    try:
        if NbaStatsClient(cfg).reachable(cfg["tonight"]["nba_probe_timeout_s"]):
            cmd_backfill(argparse.Namespace(db=args.db, seasons=None, skip_matchups=False, skip_spreads=False,
                                            refresh=True), cfg)
            return
        log.warning("tonight: stats.nba.com unreachable — ESPN schedule and spreads only; NBA game logs not "
                    "updated (settlement uses ESPN final box scores)")
        from propedge.ingest.history import ingest_historical_spreads, ingest_schedule
        hist = cfg["history"]
        today = datetime.now(EASTERN).date()
        espn = EspnLiveProvider()
        with _session_factory(args.db)() as session:
            n = ingest_schedule(session, espn, today - timedelta(days=hist["schedule_days_back"]),
                                today + timedelta(days=hist["schedule_days_ahead"]))
            session.commit()
            for season in hist["spreads_for"]:
                ingest_historical_spreads(session, espn, season)
            session.commit()
        log.info("tonight: ESPN schedule backfill %d rows", n)
    except Exception:
        log.exception("tonight: backfill failed — continuing to polling without it")


def cmd_tonight(args, cfg) -> None:
    from propedge.data.espn import EspnLiveProvider
    from propedge.data.recorder import RawRecorder
    from propedge.ingest.live import LivePoller
    from propedge.ingest.tonight import keep_awake, plan_start, utcnow, wait_until
    from propedge.odds.kalshi import KalshiOddsProvider
    import time

    t = cfg["tonight"]
    # Build the side jobs (and import everything they use) NOW: config is read once at start, and
    # importing code hours later could pick up a half-edited file or a config it doesn't match.
    jobs = _tonight_jobs(cfg, args.db)
    caffeinate = keep_awake()
    if caffeinate:
        log.info("tonight: keeping the Mac awake while this runs (caffeinate)")
    if not args.skip_backfill:
        log.info("tonight: step 1/3 backfill")
        _tonight_backfill(args, cfg)

    from propedge.ids import EASTERN
    game_day = datetime.now(EASTERN).date()      # the night this run is for (fixed, even after midnight)
    games = _todays_games(cfg, args.db, t, game_day)
    start = plan_start(games, utcnow(), timedelta(minutes=t["start_before_first_tip_min"]))
    if start is None:
        log.info("tonight: no games left today (%d scheduled) — nothing to do", len(games))
        return
    log.info("tonight: step 2/3 %d games today; polling starts %s", len(games),
             start.astimezone().strftime("%H:%M %Z"))
    wait_until(start, utcnow, time.sleep, timedelta(minutes=t["wait_log_every_min"]))

    log.info("tonight: step 3/3 polling + recording until every game is final")
    recorder = RawRecorder(ROOT / cfg["paths"]["data_dir"] / "recordings", cfg["recording"]["min_interval_s"])
    poller = LivePoller(_session_factory(args.db), EspnLiveProvider(on_raw=recorder),
                        KalshiOddsProvider(config=cfg, on_raw=recorder), cfg, THIN_SLICE_STATS)
    poller.day = game_day
    for job in jobs:
        poller.jobs.append(job)
    from propedge.edge.live import LiveEngine
    poller.engine = LiveEngine(poller.store.session, cfg)      # paper predictions + flags during games
    try:
        poller.run(stop_when_all_final=True)
    except KeyboardInterrupt:
        log.info("tonight: stopped (Ctrl-C)")
    finally:
        if caffeinate:
            caffeinate.terminate()
    cmd_status(argparse.Namespace(db=args.db, hours=12), cfg)
    # Nightly event reports: holds are minutes long, so no settlement is needed (DECISIONS #62, #65).
    for name, fn, ns in (("frozen test", cmd_rules_report, argparse.Namespace(db=args.db, show_before=False)),
                         ("exploratory event trades", cmd_event_trades,
                          argparse.Namespace(db=args.db, since=None, date=str(game_day)))):
        try:
            print(f"\n----- {name} -----")
            fn(ns, cfg)
        except Exception:
            log.exception("tonight: %s report failed", name)


def _todays_games(cfg, db_url, t, today):
    """Today's schedule (US-Eastern date asked for EXPLICITLY: early in the morning ESPN's default
    scoreboard still shows yesterday's finished games). A failed fetch must never look like "no games":
    retry ESPN, then fall back to the schedule already in the database (loaded by backfill)."""
    import time
    from sqlalchemy import select
    from propedge.data.espn import EspnLiveProvider
    from propedge.data.live_base import ScheduledGame

    espn = EspnLiveProvider()
    for attempt in range(t["scoreboard_retries"]):
        games = espn.scoreboard(today)
        if games:
            return games
        log.warning("tonight: empty scoreboard (attempt %d) — retrying in %ss", attempt + 1, t["scoreboard_retry_s"])
        time.sleep(t["scoreboard_retry_s"])
    with _session_factory(db_url)() as session:
        rows = session.scalars(select(m.Game).where(m.Game.game_date == today)).all()
    log.warning("tonight: ESPN scoreboard unavailable; using %d games from the database schedule", len(rows))
    return [ScheduledGame(g.espn_event_id or g.id, g.game_date, g.start_time.replace(tzinfo=timezone.utc)
                          if g.start_time and g.start_time.tzinfo is None else g.start_time,
                          g.away, g.home, g.status or "pre") for g in rows]


def _tonight_jobs(cfg, db_url):
    """Periodic side jobs during tonight's run: the pre-game Kalshi comparison (paper flags)
    every `edge.pregame_every_min` until every game has tipped off."""
    from propedge.data.espn import EspnLiveProvider
    from propedge.edge.pregame import run_pregame
    from propedge.ids import EASTERN
    from propedge.odds.kalshi import KalshiOddsProvider

    Session = _session_factory(db_url)
    espn, kalshi = EspnLiveProvider(), KalshiOddsProvider(config=cfg)

    def pregame_job(now: datetime) -> None:
        with Session() as session:
            res = run_pregame(session, cfg, now.astimezone(EASTERN).date(), kalshi.fetch_quotes(["pts"]),
                              espn.injuries(), now, only_not_started=True)
            if not res.projections.empty:
                log.info("pregame: %d players projected, %d would-be flags", len(res.projections), len(res.flags))

    pregame_job.__name__ = "pregame"
    return [(timedelta(minutes=cfg["edge"]["pregame_every_min"]), pregame_job)]


def cmd_backtest_points(args, cfg) -> None:
    import json
    from datetime import date
    import yaml
    from propedge.backtest.points import evaluate, load_frames, tune

    bt = cfg["backtest"]
    Session = _session_factory(args.db)
    with Session() as session:
        players, teams, games = load_frames(session, bt["seasons"])
    params = dict(cfg["model"]["points"])
    if args.tune:
        params = tune(players, teams, games, params, bt["tune_season"], date.fromisoformat(bt["tune_from"]))
        changed = {k: v for k, v in params.items() if v != cfg["model"]["points"][k]}
        print("\nTuned parameters that differ from config.yaml (copy into model.points):")
        print(yaml.safe_dump({k: (round(v, 5) if isinstance(v, float) else v) for k, v in changed.items()},
                             sort_keys=False))
    result = evaluate(players, teams, games, params, bt["test_season"], cfg["model"]["ladder_lines"])
    out = ROOT / "reports" / "phase2_backtest.json"
    out.write_text(json.dumps({"params": params, "result": result}, indent=2, default=float))
    a = result["all"]
    print(f"\nTest season {bt['test_season']}: {result['rows']} player-games")
    print(f"Bias factor applied: mean {result['bias_factor']['mean']:.3f} "
          f"(range {result['bias_factor']['min']:.3f}–{result['bias_factor']['max']:.3f})")
    for name in ("model", "model_no_bias_correction", "season_avg", "last10"):
        print(f"  {name:24} MAE {a[name]['mae']:.3f}  RMSE {a[name]['rmse']:.3f}  bias {a[name]['bias']:+.3f}")
    print(f"Details: {out.relative_to(ROOT)}")


def cmd_pregame(args, cfg) -> None:
    from datetime import date
    from propedge.data.espn import EspnLiveProvider
    from propedge.edge.pregame import actual_absences, run_pregame, settle
    from propedge.ids import EASTERN
    from propedge.ingest.live import LiveStore
    from propedge.odds.kalshi import KalshiOddsProvider, load_quotes_file

    today = datetime.now(EASTERN).date()
    day = date.fromisoformat(args.date) if args.date else today
    series = cfg["odds"]["kalshi"]["series"]["pts"]
    if args.quotes_file:
        at = datetime.fromisoformat(args.quotes_time.replace("Z", "+00:00")) if args.quotes_time else None
        quotes = load_quotes_file(args.quotes_file, "pts", series, at)
        now = max(q.fetched_at for q in quotes)
    else:
        quotes = KalshiOddsProvider(config=cfg).fetch_quotes(["pts"])
        now = datetime.now(timezone.utc)
    Session = _session_factory(args.db)
    with Session() as session:
        if not args.no_log:   # flags reference stored markets; a dry run writes nothing
            LiveStore(session, stale_after_s=cfg["live_feed"]["stale_after_s"]).store_quotes(quotes)
        finished = day < today
        injuries = [] if finished else EspnLiveProvider().injuries()
        override = actual_absences(session, cfg, day) if finished else None
        res = run_pregame(session, cfg, day, quotes, injuries, now, known_absent_override=override,
                          log_flags=not args.no_log, record=not args.no_log)
        _print_pregame(res, args.top)
        if finished and not res.flags.empty:
            settled = settle(session, res.flags, day)
            for status in ("flag", "suppressed"):
                s_ = settled[(settled["status"] == status) & (settled["result"] != "void")]
                if len(s_):
                    print(f"Paper results ({status}): {len(s_)} bets, {int((s_['result'] == 'win').sum())} won, "
                          f"P&L {s_['pnl_per_contract'].sum():+.2f} per contract each "
                          f"(avg cost {s_['cost'].mean():.2f})")
        out = ROOT / cfg["paths"]["data_dir"] / "pregame"
        out.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        res.projections.to_csv(out / f"{day}_projections_{stamp}.csv", index=False)
        res.rungs.to_csv(out / f"{day}_rungs_{stamp}.csv", index=False)
        if not res.flags.empty:
            res.flags.drop(columns=["inputs"]).to_csv(out / f"{day}_flags_{stamp}.csv", index=False)
        print(f"Saved CSVs under {out.relative_to(ROOT)}/")


def _print_pregame(res, top: int) -> None:
    import pandas as pd
    if res.projections.empty:
        print("No projections.")
        return
    pr = res.projections.sort_values("proj_points", ascending=False).head(top)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pr[["player", "team", "opponent", "proj_minutes", "proj_points", "points_sd", "confidence",
                  "low_reasons", "injury"]].round(1).to_string(index=False))
    if not res.flags.empty:
        fl = res.flags.sort_values("edge", ascending=False)
        counts = fl["status"].value_counts().to_dict()
        sides = fl[fl.status == "flag"]["side"].value_counts().to_dict()
        print(f"\nCandidates by status: {counts}")
        print(f"Flags by side: YES {sides.get('yes', 0)} / NO {sides.get('no', 0)}"
              "  (a steady lean to one side = bias, not edge)")
        fl = fl[fl.status != "below_threshold"].assign(why=fl["reasons"].map(lambda rs: "; ".join(r["code"] for r in rs)))
        print(fl[["player", "line", "side", "model_prob", "ask", "cost", "edge", "status", "why"]]
              .round(3).head(25).to_string(index=False))


def cmd_scoreboard(args, cfg) -> None:
    from datetime import date
    import pandas as pd
    from sqlalchemy import select
    from propedge.edge.scoreboard import clv_summary, flag_clv, retro_day, scoreboard, settled_rungs, side_split
    from propedge.http import make_client
    from propedge.ingest.closing import capture_from_candles, capture_from_snapshots, tip_time

    kalshi, base = make_client("kalshi"), cfg["odds"]["kalshi"]["base_url"]
    series = cfg["odds"]["kalshi"]["series"]["pts"]
    since, until = _night_range(args)
    Session = _session_factory(args.db)
    with Session() as session:
        for d in args.retro or []:
            log.info("retro %s: %s", d, retro_day(session, cfg, date.fromisoformat(d), kalshi, base))
        # Closing quotes for finished games that don't have them yet (own snapshots first, else candles).
        from propedge.ingest.closing import stale_or_missing
        for g in session.scalars(select(m.Game).where(m.Game.status == "post", m.Game.id.in_(
                select(m.PropMarket.game_id).distinct()))):
            if (since and g.game_date < since) or (until and g.game_date > until):
                continue
            todo = stale_or_missing(session, g.id)
            if not todo:
                continue
            tip = tip_time(session, g.id)
            if tip is None:
                continue
            capture_from_snapshots(session, g.id, tip)          # fresh polled quotes first…
            todo = stale_or_missing(session, g.id)
            n = capture_from_candles(session, kalshi, base, series, g.id, todo, tip) if todo else 0   # …else candles
            log.info("closing quotes for %s: %d from candles", g.id, n)

        df = settled_rungs(session, since, until)
        pd.set_option("display.width", 200)
        if df.empty:
            print("No settled contracts with both a pre-game prediction and a closing quote yet.")
        else:
            b = scoreboard(df)
            print(f"\nMODEL vs KALSHI — {len(df)} settled contracts, {df['game_id'].nunique()} games, "
                  f"{df['game_date'].nunique()} nights. Lower is better.")
            print(f"Blend weight on the model (fitted on these results): {b.blend_weight:.2f}  "
                  "(0 = trust Kalshi entirely, 1 = trust the model entirely)")
            t = b.table.copy()
            print(t.round(4).to_string(index=False))
            print(f"Prediction sources: {df['pred_source'].value_counts().to_dict()}; "
                  f"closing sources: {df['close_source'].value_counts().to_dict()}")
        clv = flag_clv(session, since, until)
        if not clv.empty:
            print("\nPRE-GAME CLV — first time each contract/side was logged vs Kalshi's closing mid")
            print("(positive = the market moved our way before tip-off)")
            print(clv_summary(clv).round(4).to_string(index=False))
        from propedge.backtest.live import score_table
        from propedge.edge.scoreboard import live_flag_results, live_rows
        lv = live_rows(session, since, until)
        if not lv.empty:
            print(f"\nLIVE (real time) — {len(lv)} predictions on {lv['game_id'].nunique()} games, each vs the Kalshi "
                  "quote current at that moment (our state ~50 s old). Lower is better.")
            print(score_table(lv, cols=("p_model", "p_kalshi")).round(4).to_string(index=False))
            normal = lv[lv["confidence"] == "normal"]
            if len(normal):
                print("Normal-confidence players only:")
                print(score_table(normal, cols=("p_model", "p_kalshi")).round(4).to_string(index=False))
        lf = live_flag_results(session, since, until)
        if not lf.empty:
            g = lf.groupby(["status", "side"])
            print("\nLIVE PAPER CANDIDATES SETTLED (first log per contract/side/status; P&L per contract after fee)")
            print(pd.DataFrame({"n": g.size(), "win_rate": g["won"].mean(), "avg_cost": g["cost"].mean(),
                                "avg_pnl": g["pnl"].mean(), "avg_edge_claimed": g["edge"].mean()})
                  .reset_index().round(3).to_string(index=False))
        split = side_split(session, since, until)
        if not split.empty:
            print("\nFLAGS BY SIDE PER NIGHT (a steady lean = bias, not edge)")
            print(split.to_string())


def cmd_backfill_pbp(args, cfg) -> None:
    from datetime import date
    from propedge.data.espn import EspnLiveProvider
    from propedge.ingest.history import ingest_playbyplay

    Session = _session_factory(args.db)
    with Session() as session:
        n = ingest_playbyplay(session, EspnLiveProvider(), date.fromisoformat(args.start), date.fromisoformat(args.end))
    log.info("play-by-play backfill: %d games stored", n)


def cmd_edge_test(args, cfg) -> None:
    import pandas as pd
    from propedge.backtest.repricing import run, summarize
    from propedge.http import make_client

    Session = _session_factory(args.db)
    with Session() as session:
        df = run(session, cfg, make_client("kalshi"), ROOT / cfg["paths"]["data_dir"] / "cache" / "kalshi_trades",
                 source=args.source)
    out = ROOT / "reports" / f"phase3_repricing_events_{args.source}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(f"{len(df)} measurable moments across {df['game_id'].nunique() if len(df) else 0} games "
          "(placebo = same contracts 15 min before an event, nothing happening)")
    print("Among moments with a clear move (≥5¢ between −4..−1 min and +5..+15 min): median share of that move "
          "already done at each delay after the event:")
    print(summarize(df, cfg["live"]["assumed_feed_delay_s"]).round(2).to_string(index=False))
    print(f"Per-event details: {out.relative_to(ROOT)}")


def cmd_backtest_live(args, cfg) -> None:
    import json
    import yaml
    import pandas as pd
    from propedge.backtest.live import game_ids, load_game, pregame_table, run_games, score_table, tune_live

    lb = cfg["live_backtest"]
    p = {**cfg["model"]["points"], **cfg["live"]}
    Session = _session_factory(args.db)
    with Session() as session:
        pre = pregame_table(session, cfg, [lb["season"]])
        tune_ids = game_ids(session, lb["season"], end=lb["tune_until"])
        test_ids = game_ids(session, lb["season"], start=lb["test_from"])
        if args.max_games:
            tune_ids, test_ids = tune_ids[: args.max_games], test_ids[: args.max_games]
        if args.tune:
            tune_games = [g for g in (load_game(session, i) for i in tune_ids[:: lb["tune_every_nth_game"]]) if g]
            log.info("tuning on %d games", len(tune_games))
            p = tune_live(tune_games, pre, p, n_sims=lb["tune_sims"])
            changed = {k: p[k] for k in cfg["live"] if p[k] != cfg["live"][k]}
            print("Tuned live parameters that differ from config.yaml (copy into live:):")
            print(yaml.safe_dump(changed, sort_keys=False))
        test_games = [g for g in (load_game(session, i) for i in test_ids) if g]
    df = run_games(test_games, pre, p, cfg["live"]["n_sims"] if not args.fast else 1000)
    out = ROOT / "reports" / "phase3_live_backtest.json"
    by_q = score_table(df)
    by_q_normal = score_table(df[~df["low_conf"]])
    out.write_text(json.dumps({"games": len(test_games), "rows": len(df), "by_quarter": by_q.to_dict("records"),
                               "by_quarter_normal_conf": by_q_normal.to_dict("records")}, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(f"\nLive backtest on {len(test_games)} games ({lb['test_from']} on), {len(df)} open ladder lines, "
          f"state delayed {cfg['live']['assumed_feed_delay_s']}s. Lower is better.")
    print(by_q.round(4).to_string(index=False))
    print("\nNormal-confidence players only:")
    print(by_q_normal.round(4).to_string(index=False))

    # Model vs Kalshi on this season's games: same moments, Kalshi's mid from 1-minute candles.
    from propedge.backtest.live import CandleMids
    from propedge.http import make_client
    with Session() as session:
        cur = lb["kalshi_season"]
        pre_k = pregame_table(session, cfg, [cur])
        mids = CandleMids(session, make_client("kalshi"), cfg, ROOT / cfg["paths"]["data_dir"] / "cache" / "kalshi_trades")
        kgames = [g for g in (load_game(session, i) for i in game_ids(session, cur)) if g]
        rows = []
        for g in kgames:
            rows.append(run_games([g], pre_k, p, cfg["live"]["n_sims"] if not args.fast else 1000,
                                  kalshi_mid=mids.for_game(g)))
    dk = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not dk.empty:
        dk = dk[dk["p_kalshi"].notna()]
        kq = score_table(dk)
        print(f"\nMODEL vs KALSHI live, {cur}: {len(kgames)} games, {len(dk)} contract-moments "
              "(Kalshi mid from 1-minute candles at the same real time; our state 50 s older).")
        print(kq.round(4).to_string(index=False))
        doc = json.loads(out.read_text())
        doc["vs_kalshi"] = kq.to_dict("records")
        out.write_text(json.dumps(doc, indent=2, default=float))


def cmd_replay_live(args, cfg) -> None:
    """Run the live engine over recorded nights exactly as they were seen (ESPN states with their real
    delay, Kalshi quotes recorded at the time) and score it against Kalshi's mid, by quarter."""
    import json
    import pandas as pd
    from propedge.backtest.live import score_table
    from propedge.data.espn import parse_summary
    from propedge.data.replay import _Recording, _load
    from propedge.edge.live import LiveEngine
    from propedge.ids import game_key
    from propedge.models.suppression import QuotePoint
    from propedge.odds.kalshi import parse_market

    dirs = [Path(d) for d in args.dirs] if args.dirs else sorted((ROOT / cfg["paths"]["data_dir"] / "recordings").iterdir())
    Session = _session_factory(args.db)
    stale_after = cfg["live_feed"]["stale_after_s"]
    all_rows = []
    with Session() as session:
        for d in dirs:
            rec = _Recording(d)
            kal = rec.files.get(("kalshi", cfg["odds"]["kalshi"]["series"]["pts"]), [])
            kal_quotes = []           # [(time, [PropQuote])] in time order, parsed lazily
            for t, path in kal:
                fetched, data = _load(path)
                kal_quotes.append((fetched, [q for q in (parse_market(mk, "pts", fetched) for mk in data["markets"]) if q]))
            engine = LiveEngine(session, cfg, is_mock=True, log_flags=False, record=False, source="replay")
            for gid_espn in rec.keys("espn"):
                for t, path in rec.files[("espn", gid_espn)]:
                    fetched, data = _load(path)
                    state = parse_summary(data, fetched)
                    gid = game_key(state.game_date, state.away, state.home)
                    before = [qs for ft, qs in kal_quotes if ft <= fetched]
                    if not before:
                        continue
                    quotes = [q for q in before[-1] if q.game_date == state.game_date and q.home == state.home]
                    history: dict[str, list] = {}
                    for ft, qs in kal_quotes:
                        if fetched - pd.Timedelta(minutes=6) <= ft <= fetched:
                            for q in qs:
                                history.setdefault(q.ticker, []).append(QuotePoint(ft, q.yes_bid, q.yes_ask))
                    lag = state.feed_lag_s
                    res = engine.on_state(state, gid, fetched, bool(lag is not None and lag > stale_after),
                                          quotes=quotes, history=history)
                    if not res.empty:
                        all_rows.append(res.assign(game_id=gid, period=state.period, night=d.name))
        df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        if df.empty:
            print("No live predictions could be made from the recordings.")
            return
        from propedge.edge.scoreboard import final_points
        fp = final_points(session, df["game_id"].unique().tolist())
        finals = {(str(pid), gid): pts for pid, gid, pts in fp.itertuples(index=False)}
    df["final"] = [finals.get((str(p), g)) for p, g in zip(df["player_id"], df["game_id"])]
    df = df[df["final"].notna()].copy()
    df["y"] = (df["final"] > df["line"]).astype(float)
    df["p_model"] = df["p_yes"]
    df["p_kalshi"] = [((b + a) / 2) if b is not None and a is not None and b == b and a == a else None
                      for b, a in zip(df["yes_bid"], df["yes_ask"])]
    df = df[df["p_kalshi"].notna()]
    df["quarter"] = df["period"].clip(upper=4)
    t = score_table(df, cols=("p_model", "p_kalshi"))
    out = ROOT / "reports" / "phase3_replay_live.json"
    out.write_text(json.dumps({"nights": [d.name for d in dirs], "games": int(df["game_id"].nunique()),
                               "rows": len(df), "by_quarter": t.to_dict("records")}, indent=2, default=float))
    print(f"Replayed {len(dirs)} recorded night(s), {df['game_id'].nunique()} games, {len(df)} contract-moments "
          "(our real delayed feed vs Kalshi's quote at the same moment). Lower is better.")
    print(t.round(4).to_string(index=False))


def cmd_replay_events(args, cfg) -> None:
    """Run the live engine's event triggers over polled nights stored in the database (as seen at the time)."""
    from datetime import date
    from sqlalchemy import select
    from propedge.edge.live import LiveEngine
    from propedge.ingest.db_replay import states_from_db

    Session = _session_factory(args.db)
    with Session() as session:
        from sqlalchemy import delete
        for d in args.dates:
            day = date.fromisoformat(d)
            gids = [g.id for g in session.scalars(select(m.Game).where(m.Game.game_date == day))]
            # re-running a night replaces its earlier replayed event flags
            session.execute(delete(m.Flag).where(m.Flag.kind == "event", m.Flag.is_mock.is_(True),
                                                 m.Flag.game_id.in_(gids)))
            session.commit()
            engine = LiveEngine(session, cfg, is_mock=True, log_flags=True, record=False, source="replay")
            engine.log_routine = False
            for gid in gids:
                n = 0
                for snap, state in states_from_db(session, gid):
                    engine.on_state(state, gid, snap.fetched_at.replace(tzinfo=timezone.utc)
                                    if snap.fetched_at.tzinfo is None else snap.fetched_at, snap.is_stale)
                    n += 1
                if n:
                    log.info("replayed %s: %d states", gid, n)
            k = session.query(m.Flag).filter(m.Flag.kind == "event", m.Flag.game_id.in_(gids)).count()
            log.info("%s: %d event candidates logged", d, k)


def cmd_event_report(args, cfg) -> None:
    from datetime import date
    import pandas as pd
    from propedge.edge.events import event_rows, event_summary

    since, until = _night_range(args)
    Session = _session_factory(args.db)
    with Session() as session:
        df = event_rows(session, cfg, since, include_replay=not args.live_only, until=until)
    if df.empty:
        print("No event candidates yet.")
        return
    out = ROOT / "reports" / "events_candidates.csv"
    df.to_csv(out, index=False)
    hz = cfg["events"]["clv_horizons_s"]
    pd.set_option("display.width", 250)
    print(f"EVENT CANDIDATES: {len(df)} on {df['game_id'].nunique()} games "
          f"({int(df['replay'].sum())} from replays of recorded nights, {int((~df['replay']).sum())} live)")
    print("move = our side's mid h minutes later − mid at the event (did Kalshi come our way?)")
    print("CLV  = our side's mid h minutes later − the ASK we'd have paid (also covers crossing the spread)")
    print("\nBy event, role and side (all logged candidates):")
    print(event_summary(df, hz).round(3).to_string(index=False))
    print("\nBy status (flag = passed event thresholds, not suppressed):")
    print(event_summary(df, hz, by=("status", "side")).round(3).to_string(index=False))
    with Session() as session:
        base = event_rows(session, cfg, since, include_replay=False, kind="live", until=until)
    if not base.empty:
        base = base[base["game_id"].isin(df["game_id"].unique())]
        print("\nBASELINE — routine live candidates on the same games at ordinary moments (no event):")
        print(event_summary(base.assign(type="routine", role="-"), hz).round(3).to_string(index=False))
    print(f"\nPer-candidate details: {out.relative_to(ROOT)}")


def _night_range(args) -> tuple:
    """--date D = that night only; --since D = that night and later (US-Eastern game dates)."""
    from datetime import date
    if getattr(args, "date", None):
        d = date.fromisoformat(args.date)
        return d, d
    return (date.fromisoformat(args.since) if getattr(args, "since", None) else None), None


def _fee_check(cfg) -> str:
    """Kalshi's current fee type for points vs what config assumes (maker fee or not)."""
    from propedge.odds.kalshi import KalshiOddsProvider
    want = cfg["odds"]["kalshi"]["fees"]["fee_type"]["pts"]
    try:
        info = KalshiOddsProvider(config=cfg).series_fee_info("pts")
    except Exception:       # offline: the report still runs on config
        info = None
    if not info:
        return f"fee type: {want} (from config; Kalshi not reachable to re-check)"
    if info["fee_type"] != want or info["fee_multiplier"] != 1:
        return (f"WARNING: Kalshi now reports fee_type={info['fee_type']} multiplier={info['fee_multiplier']} for "
                f"points, config assumes {want} x1. Update odds.kalshi.fees.fee_type and re-run.")
    return f"fee type: {want} x1 (confirmed with Kalshi's API just now) — taker fee both ways, maker fee 0"


def cmd_event_trades(args, cfg) -> None:
    """EXPLORATORY: buy shortly after an event, sell 1/2/5 minutes later, by independent event."""
    from datetime import date
    import numpy as np
    import pandas as pd
    from propedge.edge.event_trades import by_event, game_ids_for, load_games, model_picks, simulate
    from propedge.models.trade_sim import Fees

    since, until = _night_range(args)
    since = since or date(2026, 10, 23)      # first night with ~3 s Kalshi quotes
    fr, et = cfg["frozen_rules"], cfg["event_trades"]
    fees = Fees.from_config(cfg)
    Session = _session_factory(args.db)
    with Session() as session:
        gids = game_ids_for(session, since, until)
        games = list(load_games(session, gids, fr["blowout_lead"], fr["blowout_from_min"], fr["foul_n"]))
        trades = simulate(games, cfg, fees, model_picks(session, gids))
    if trades.empty:
        print("No events found.")
        return
    out = ROOT / "reports" / "event_trades.csv"
    trades.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    n_ev = trades.groupby("type")["event"].nunique().to_dict()
    nights = sorted({str(d) for d in trades["game_date"]})
    print(f"EXPLORATORY — not the frozen test (see `rules-report`). Nights: {', '.join(nights)}; "
          f"{len(games)} games; independent events: {n_ev}")
    print(_fee_check(cfg))
    print("net = exit price − exit fee − (entry ask + entry fee), per contract, averaged within each event, then "
          f"across events; {int(et['bootstrap']['level'] * 100)}% interval by event. Sides: fouler NO, starters NO.")
    print("selection: model = the engine's best-edge event candidate for that player (its side; none if it logged "
          "nothing) | closest50 = rung with YES mid nearest 50¢ among spreads ≤ "
          f"{fr['rung_max_spread'] * 100:.0f}¢ (as in the frozen rules) | closest50_any = same, any spread | "
          "all = every rung with YES mid 5–95¢, averaged")
    t = trades.assign(subset=np.where(trades["type"] == "blowout", "blowout → starters NO",
                                      np.where(trades["period"] <= 3, "4th foul Q1–Q3 → fouler NO",
                                               "4th foul Q4/OT → fouler NO")))
    b = et["bootstrap"]
    s = by_event(t, ["subset", "selection", "delay_s", "hold_s", "exit"], b["level"], b["n"], b["seed"],
                 cfg["odds"]["kalshi"]["order_contracts"])
    cols = ["events", "trades", "gross", "fees", "net", "ci_lo", "ci_hi", "events_positive"]
    for sub, d in s.groupby("subset", sort=False):
        print(f"\n=== {sub} ===")
        print("Exit: sell at the bid (taker fee)")
        print(d[d["exit"] == "bid"][["selection", "delay_s", "hold_s"] + cols + ["ask_ge_order"]]
              .round(3).to_string(index=False))
        print("Exit: resting limit sell 1¢ inside the spread, filled only if a recorded bid reaches it within "
              f"{et['limit_wait_s']} s, else the bid then (maker fee 0 for this series)")
        print(d[d["exit"] == "limit"][["selection", "delay_s", "hold_s"] + cols + ["limit_fill"]]
              .round(3).to_string(index=False))
    missing = trades[trades.get("pnl").isna()] if "pnl" in trades else trades
    if len(missing):
        print("\nTrades that couldn't be simulated:", missing["reason"].value_counts().to_dict())
    print(f"\nPer-trade details: {out.relative_to(ROOT)}")


def cmd_watch(args, cfg) -> None:
    from datetime import date
    from propedge.watch import run
    try:
        run(cfg, args.every, date.fromisoformat(args.date) if args.date else None, args.once)
    except KeyboardInterrupt:
        pass


def cmd_rules_report(args, cfg) -> None:
    """The frozen test (DECISIONS #62): Rules A and B, nights from frozen_rules.judge_from only."""
    from datetime import date, timedelta
    import pandas as pd
    from propedge.edge.event_trades import frozen_summary, frozen_trades, game_ids_for, load_games
    from propedge.models.trade_sim import Fees

    fr = cfg["frozen_rules"]
    start = date.fromisoformat(fr["judge_from"])
    fees = Fees.from_config(cfg)
    Session = _session_factory(args.db)

    def run(since, until):
        with Session() as session:
            games = load_games(session, game_ids_for(session, since, until), fr["blowout_lead"],
                               fr["blowout_from_min"], fr["foul_n"])
            return frozen_trades(games, cfg, fees)

    pd.set_option("display.width", 250)
    cols = ["detected", "events", "trades", "gross", "fees", "net", "ci_lo", "ci_hi", "events_positive",
            "pnl_10_contracts", "ask_ge_order"]
    print("FROZEN TEST (DECISIONS #62) — rules unchanged since 2026-10-25; counts nights from "
          f"{fr['judge_from']}. Entry {fr['entry_delay_s']} s after our data shows the event, at the ask; "
          "exit at the bid; taker fee both ways.")
    print("  Rule A: 4th foul before Q4 → his under, rung nearest 50¢, sell after 5 min")
    print("  Rule B: blowout starts → starters' unders, rung nearest 50¢, sell after 2 min")
    print(f"  Rung nearest 50¢ = YES mid nearest 0.50 among rungs with spread ≤ {fr['rung_max_spread'] * 100:.0f}¢ "
          "(no such rung = event reported, not traded, not counted)")
    print(_fee_check(cfg))
    tr = run(start, None)
    if tr.empty or "pnl" not in tr:
        print(f"\nNo counted events yet (first counted night: {fr['judge_from']}).")
    else:
        tr.to_csv(ROOT / "reports" / "frozen_rules_trades.csv", index=False)
        nightly = frozen_summary(tr, cfg, by_night=True)
        total = frozen_summary(tr, cfg, by_night=False)
        print("\nBy night:")
        print(nightly[["rule", "game_date"] + [c for c in cols if c in nightly]].round(3).to_string(index=False))
        print("\nCumulative:")
        print(total[["rule"] + [c for c in cols if c in total]].round(3).to_string(index=False))
        a = total[total["rule"] == "rule_a"]
        n_a = int(a["events"].iloc[0]) if len(a) and a["events"].notna().all() else 0
        print(f"\nDecision point (DECISIONS #61): Rule A has {n_a} of {fr['decision_events']} independent events.")
        if (tr.get("pnl") is not None) and tr["pnl"].isna().any():
            print("Not simulated:", tr[tr["pnl"].isna()]["reason"].value_counts().to_dict())
        print("Per-trade details: reports/frozen_rules_trades.csv")
    if args.show_before:
        old = run(None, start - timedelta(days=1))
        if not old.empty and "pnl" in old:
            print("\nFOR REFERENCE ONLY — nights before the freeze. These shaped the rules, so they DON'T count:")
            s = frozen_summary(old, cfg, by_night=True)
            print(s[["rule", "game_date"] + [c for c in cols if c in s]].round(3).to_string(index=False))


def cmd_replay(args, cfg) -> None:
    from propedge.data.replay import ReplayClock, ReplayLiveProvider, ReplayOddsProvider, _Recording
    from propedge.ingest.live import LivePoller

    directory = Path(args.directory)
    rec = _Recording(directory)
    clock = ReplayClock(rec.start)
    live = ReplayLiveProvider(directory, clock)
    odds = ReplayOddsProvider(directory, clock, cfg["odds"]["kalshi"]["series"])
    db = args.db or REPLAY_DB
    log.info("REPLAY (mock data) %s -> %s, %s to %s", directory, db, rec.start, rec.end)
    poller = LivePoller(_session_factory(db), live, odds, cfg, THIN_SLICE_STATS, now=clock.now, sleep=clock.sleep)
    poller.run(max_minutes=(rec.end - rec.start).total_seconds() / 60 + 3)


def cmd_status(args, cfg) -> None:
    Session = _session_factory(args.db)
    with Session() as s:
        print("Rows:")
        for model in (m.Team, m.Player, m.Game, m.PlayerGameLog, m.TeamGameLog, m.TeamSeasonStats, m.Matchup,
                      m.LiveGameSnapshot, m.LivePlayerSnapshot, m.Play, m.PropMarket, m.OddsSnapshot):
            print(f"  {model.__tablename__:22} {s.scalar(select(func.count()).select_from(model)):>8}")

        since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
        plays = s.execute(select(m.Play.first_seen_at, m.Play.wallclock).where(
            m.Play.wallclock.is_not(None), m.Play.first_seen_at >= since, m.Play.backlog.is_(False))).all()
        delays = [(utc(f) - utc(w)).total_seconds() for f, w in plays]
        if delays:
            q = statistics.quantiles(delays, n=10)
            print(f"\nFeed delay over the last {args.hours}h (time from a play happening to us storing it), "
                  f"{len(delays)} plays: median {statistics.median(delays):.0f}s, "
                  f"p10 {q[0]:.0f}s, p90 {q[-1]:.0f}s")
        snaps = s.execute(select(m.LiveGameSnapshot.is_stale).where(m.LiveGameSnapshot.fetched_at >= since,
                                                                    m.LiveGameSnapshot.status == "in")).all()
        if snaps:
            print(f"Live snapshots marked stale (> {cfg['live_feed']['stale_after_s']}s): "
                  f"{sum(x[0] for x in snaps)}/{len(snaps)}")

        markets = s.scalar(select(func.count()).select_from(m.PropMarket))
        unmatched = s.execute(select(m.PropMarket.player_name, m.PropMarket.team).where(
            m.PropMarket.player_id.is_(None)).distinct()).all()
        if markets:
            print(f"\nOdds markets: {markets}; players not matched to our database: {len(unmatched)}"
                  + (f" -> {', '.join(f'{n} ({t})' for n, t in unmatched[:15])}" if unmatched else ""))
        latest = s.scalar(select(func.max(m.OddsSnapshot.last_seen_at)))
        if latest:
            print(f"Latest odds confirmation: {latest} UTC")
        mock = s.scalar(select(func.count()).select_from(m.LiveGameSnapshot).where(m.LiveGameSnapshot.is_mock))
        if mock:
            print(f"\n*** This database contains {mock} MOCK/REPLAY live snapshots ***")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="propedge", description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", help="database URL (default: config paths.database_url)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")
    b = sub.add_parser("backfill")
    b.add_argument("--seasons", nargs="+", help='e.g. 2025-26 2026-27 (default: config history.seasons)')
    b.add_argument("--skip-matchups", action="store_true")
    b.add_argument("--skip-spreads", action="store_true", help="skip historical spreads (1 ESPN request per game)")
    b.add_argument("--refresh", action="store_true", help="ignore the local cache for current-season data")
    p = sub.add_parser("poll")
    p.add_argument("--minutes", type=float, help="stop after this many minutes")
    p.add_argument("--until-idle", action="store_true", help="stop when no game is live")
    p.add_argument("--no-odds", action="store_true")
    p.add_argument("--no-engine", action="store_true", help="collect data only (no live paper predictions)")
    p.add_argument("--record", action="store_true", help="also save raw payloads for `replay` (~150 MB/night)")
    tn = sub.add_parser("tonight", help="backfill, wait for tip-off, poll + record until all games end")
    tn.add_argument("--skip-backfill", action="store_true")
    bp = sub.add_parser("backtest-points", help="tune (optional) and backtest the pre-game points model")
    bp.add_argument("--tune", action="store_true", help="re-tune parameters on the tune season first")
    pg = sub.add_parser("pregame", help="project a day's players, compare with Kalshi, log paper flags")
    pg.add_argument("--date", help="US-Eastern game date (default today)")
    pg.add_argument("--quotes-file", help="use saved Kalshi quotes (recording .json.gz or raw API page)")
    pg.add_argument("--quotes-time", help="when a raw quotes file was taken (ISO UTC), if it doesn't say")
    pg.add_argument("--no-log", action="store_true", help="don't store would-be flags")
    pg.add_argument("--top", type=int, default=30)
    sb = sub.add_parser("scoreboard", help="model vs Kalshi (Brier/log loss), pre-game CLV, YES/NO split")
    sb.add_argument("--since", help="only games on/after this night (US-Eastern game date)")
    sb.add_argument("--date", help="that night's games only")
    sb.add_argument("--retro", nargs="+", metavar="DATE",
                    help="rebuild past nights first (predictions as-of that morning, closing from Kalshi candles)")
    bpp = sub.add_parser("backfill-pbp", help="ESPN play-by-play for finished games in a date range")
    bpp.add_argument("start")
    bpp.add_argument("end")
    et = sub.add_parser("edge-test", help="how fast Kalshi reprices after fouls, blowouts and exits")
    et.add_argument("--source", choices=["trades", "candles", "quotes"], default="trades",
                    help="Kalshi trade history, 1-minute bid/ask candles, or our own ~3 s quotes")
    bl = sub.add_parser("backtest-live", help="score the live engine on past games rebuilt from play-by-play")
    bl.add_argument("--tune", action="store_true")
    bl.add_argument("--fast", action="store_true", help="1,000 simulations instead of the configured count")
    bl.add_argument("--max-games", type=int)
    rl = sub.add_parser("replay-live", help="run the live engine over recorded nights; Brier vs Kalshi by quarter")
    rl.add_argument("dirs", nargs="*", help="recording directories (default: all under data/recordings)")
    rev = sub.add_parser("replay-events", help="run event triggers over polled nights stored in the database")
    rev.add_argument("dates", nargs="+", help="US-Eastern game dates, e.g. 2026-10-23")
    er = sub.add_parser("event-report", help="event flags: 1/2/5-minute CLV, tradability, results")
    er.add_argument("--since")
    er.add_argument("--live-only", action="store_true", help="exclude replayed nights")
    er.add_argument("--date", help="that night only (US-Eastern game date)")
    etr = sub.add_parser("event-trades", help="EXPLORATORY: buy after events, sell 1/2/5 min later, by event")
    etr.add_argument("--since", help="nights from this date (default 2026-10-23)")
    etr.add_argument("--date", help="that night only")
    rr = sub.add_parser("rules-report", help="the frozen test: Rules A and B (DECISIONS #62), nightly")
    rr.add_argument("--show-before", action="store_true", help="also show pre-freeze nights (reference only)")
    w = sub.add_parser("watch", help="live READ-ONLY view of tonight's run (games, lag, gaps, events, Rule A/B)")
    w.add_argument("--every", type=float, default=5.0, help="seconds between refreshes (default 5)")
    w.add_argument("--date", help="night to show (US-Eastern game date; default: the one being played)")
    w.add_argument("--once", action="store_true", help="print one screen and exit")
    r = sub.add_parser("replay")
    r.add_argument("directory", help="recording directory, e.g. data/recordings/2026-10-22")
    st = sub.add_parser("status")
    st.add_argument("--hours", type=float, default=24)

    args = parser.parse_args(argv)
    if args.command == "watch":       # read-only: no log file handler, so it never writes or rotates the log
        logging.basicConfig(level=logging.WARNING)
    else:
        setup_logging(args.verbose)
    cfg = load_config()
    {"init-db": cmd_init_db, "backfill": cmd_backfill, "poll": cmd_poll, "tonight": cmd_tonight, "backtest-points": cmd_backtest_points, "pregame": cmd_pregame, "scoreboard": cmd_scoreboard, "backfill-pbp": cmd_backfill_pbp, "edge-test": cmd_edge_test, "backtest-live": cmd_backtest_live, "replay-live": cmd_replay_live, "replay-events": cmd_replay_events,
     "event-report": cmd_event_report, "event-trades": cmd_event_trades, "rules-report": cmd_rules_report,
     "watch": cmd_watch,
     "replay": cmd_replay, "status": cmd_status}[args.command](args, cfg)


if __name__ == "__main__":
    main()
