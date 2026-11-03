"""Phase 0 smoke test: can we reach every data source, and what comes back?

Run from the project root:
    .venv/bin/python scripts/smoke_test.py

It makes a handful of requests (well under every rate limit), prints a PASS/FAIL table,
writes the details to reports/smoke_results.json, and saves raw responses under
data/samples/ so Phase 1 can be developed against real payloads.

The Odds API test only runs when asked for (`--only odds_api`) and ODDS_API_KEY is set
in .env; it spends at most 2 credits of the free 500/month.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from propedge.config import ROOT, get_secret, load_config, resolve_path  # noqa: E402
from propedge.http import SOURCE_HEADERS, FetchResult, make_client  # noqa: E402

CONFIG = load_config()
SAMPLES = resolve_path(CONFIG["paths"]["samples_dir"])
RESULTS_PATH = ROOT / "reports" / "smoke_results.json"

# First game of the 2026-27 regular season (NBA ids: 002 = regular season, 26 = season, 00001 = game #).
# Only used if the live scoreboard can't give us a recent game id.
FALLBACK_NBA_GAME_ID = "0022600001"

results: list[dict[str, Any]] = []
context: dict[str, Any] = {}


def scrub(text: str) -> str:
    """Remove API keys from URLs/messages before anything is printed or saved."""
    return re.sub(r"(apiKey=)[^&\s]+", r"\1***", text)


def save_sample(name: str, data: Any) -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    (SAMPLES / f"{name}.json").write_text(json.dumps(data, indent=1, default=str))


def record(name: str, source: str, status: str, detail: str, latency_s: float | None = None,
           extra: dict | None = None) -> None:
    results.append({
        "test": name, "source": source, "status": status, "detail": scrub(detail),
        "latency_s": None if latency_s is None else round(latency_s, 2),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **({"extra": extra} if extra else {}),
    })
    print(f"  [{status:4}] {name}: {scrub(detail)}")


def run(name: str, source: str, fn: Callable[[], tuple[str, str, dict | None]]) -> None:
    start = time.perf_counter()
    try:
        status, detail, extra = fn()
    except Exception as exc:  # a smoke test should report failures, not crash
        status, detail, extra = "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}", None
    record(name, source, status, detail, time.perf_counter() - start, extra)


def fail_detail(r: FetchResult) -> str:
    return f"{r.error} (after {r.attempts} attempt(s))"


def seconds_since(ts: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds()


# --------------------------------------------------------------------------- NBA live (cdn.nba.com)

def nba_live_raw() -> tuple[str, str, dict | None]:
    client = make_client("nba_live")
    r = client.get_json("https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json")
    if not r.ok:
        return "FAIL", fail_detail(r), None
    games = r.data["scoreboard"]["games"]
    save_sample("nba_live_scoreboard", r.data)
    finished = [g for g in games if g["gameStatus"] == 3]
    live = [g for g in games if g["gameStatus"] == 2]
    if live or finished:
        context["nba_game_id"] = (live or finished)[0]["gameId"]
    return "PASS", f"{len(games)} games today ({len(live)} live, {len(finished)} final)", None


def nba_api_live(endpoint: str) -> Callable[[], tuple[str, str, dict | None]]:
    def _run() -> tuple[str, str, dict | None]:
        from nba_api.live.nba.endpoints import boxscore, playbyplay, scoreboard

        kw = dict(headers=SOURCE_HEADERS["nba_live"], timeout=CONFIG["sources"]["nba_live"]["timeout_s"])
        game_id = context.get("nba_game_id", FALLBACK_NBA_GAME_ID)
        if endpoint == "scoreboard":
            d = scoreboard.ScoreBoard(**kw).get_dict()
            return "PASS", f"nba_api ScoreBoard OK, {len(d['scoreboard']['games'])} games", None
        if endpoint == "boxscore":
            d = boxscore.BoxScore(game_id, **kw).get_dict()
            save_sample("nba_live_boxscore", d)
            home = d["game"]["homeTeam"]
            p = home["players"][0]
            return "PASS", (f"game {game_id}: {home['teamTricode']} {len(home['players'])} players; e.g. "
                            f"{p['name']} min={p['statistics']['minutes']} pf={p['statistics']['foulsPersonal']}"), None
        d = playbyplay.PlayByPlay(game_id, **kw).get_dict()
        save_sample("nba_live_pbp", d)
        actions = d["game"]["actions"]
        return "PASS", f"game {game_id}: {len(actions)} actions; last timeActual={actions[-1].get('timeActual')}", None
    return _run


# --------------------------------------------------------------------------- NBA stats (stats.nba.com)

def nba_stats(endpoint: str) -> Callable[[], tuple[str, str, dict | None]]:
    def _run() -> tuple[str, str, dict | None]:
        from nba_api.stats.endpoints import boxscorematchupsv3, leaguegamelog

        kw = dict(timeout=CONFIG["sources"]["nba_stats"]["timeout_s"])  # nba_api's default headers
        if endpoint == "matchups":
            game_id = context.get("nba_game_id", FALLBACK_NBA_GAME_ID)
            d = boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id, **kw).get_dict()
            save_sample("nba_stats_matchups", d)
            away = d["boxScoreMatchups"]["awayTeam"]["players"]
            first = away[0]
            m = first["matchups"][0] if first.get("matchups") else None
            detail = f"game {game_id}: {len(away)} away players with matchup rows"
            if m:
                detail += (f"; e.g. {first['nameI']} guarded by {m['nameI']} for "
                           f"{m['statistics']['matchupMinutes']} min")
            return "PASS", detail, None
        d = leaguegamelog.LeagueGameLog(season="2025-26", player_or_team_abbreviation="P", **kw).get_dict()
        rows = d["resultSets"][0]["rowSet"]
        return "PASS", f"LeagueGameLog 2025-26: {len(rows)} player-game rows", None
    return _run


# --------------------------------------------------------------------------- ESPN (fallback feed)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def espn_scoreboard() -> tuple[str, str, dict | None]:
    client = make_client("espn")
    today = datetime.now().strftime("%Y%m%d")
    r = client.get_json(f"{ESPN}/scoreboard")
    if not r.ok:
        return "FAIL", fail_detail(r), None
    save_sample("espn_scoreboard", r.data)
    events = r.data["events"]
    live = [e for e in events if e["status"]["type"]["state"] == "in"]
    if live:
        context["espn_live_id"] = live[0]["id"]
    # Most recent finished game: look at today, then yesterday.
    finished = [e for e in events if e["status"]["type"]["completed"]]
    if not finished:
        yesterday = datetime.fromtimestamp(time.time() - 86400).strftime("%Y%m%d")
        r2 = client.get_json(f"{ESPN}/scoreboard", params={"dates": yesterday})
        finished = [e for e in (r2.data or {}).get("events", []) if e["status"]["type"]["completed"]]
    if finished:
        context["espn_final_id"] = finished[-1]["id"]
        context["espn_final_name"] = finished[-1]["name"]
    return "PASS", f"{len(events)} games on {today} local ({len(live)} live)", None


def espn_summary(which: str) -> Callable[[], tuple[str, str, dict | None]]:
    def _run() -> tuple[str, str, dict | None]:
        event_id = context.get("espn_live_id" if which == "live" else "espn_final_id")
        if not event_id:
            return "SKIP", f"no {which} game available right now", None
        client = make_client("espn")
        r = client.get_json(f"{ESPN}/summary", params={"event": event_id})
        if not r.ok:
            return "FAIL", fail_detail(r), None
        save_sample(f"espn_summary_{which}", r.data)
        team = r.data["boxscore"]["players"][0]
        labels = team["statistics"][0]["labels"]
        plays = r.data.get("plays", [])
        detail = (f"event {event_id}: box score cols={','.join(labels)}; {len(plays)} plays; "
                  f"spread in pickcenter={'yes' if r.data.get('pickcenter') else 'no'}")
        extra = None
        if which == "live" and plays:
            lag = seconds_since(plays[-1]["wallclock"])
            extra = {"feed_lag_s": round(lag, 1)}
            detail += f"; newest play is {lag:.0f}s old"
        return "PASS", detail, extra
    return _run


def espn_injuries() -> tuple[str, str, dict | None]:
    r = make_client("espn").get_json(f"{ESPN}/injuries")
    if not r.ok:
        return "FAIL", fail_detail(r), None
    save_sample("espn_injuries", r.data)
    teams = r.data.get("injuries", [])
    n = sum(len(t.get("injuries", [])) for t in teams)
    return "PASS", f"{n} injury entries across {len(teams)} teams", None


def nba_injury_report() -> tuple[str, str, dict | None]:
    # The official report is a PDF on an nba.com CDN; we only check reachability here.
    client = make_client("nba_live")
    url = "https://official.nba.com/nba-injury-report-2026-27-season/"
    t0 = time.perf_counter()
    resp = client.session.get(url, timeout=client.timeout_s)
    ms = time.perf_counter() - t0
    if resp.status_code != 200:
        return "FAIL", f"HTTP {resp.status_code} from official.nba.com injury page ({ms:.1f}s)", None
    return "PASS", "official injury report page reachable", None


# --------------------------------------------------------------------------- Odds

# Every NBA player-prop series, not just the ones config.yaml currently models.
KALSHI_PROP_SERIES = ["KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBAPRA",
                      "KXNBAPR", "KXNBAPA", "KXNBARA"]


def kalshi_props() -> tuple[str, str, dict | None]:
    client = make_client("kalshi")
    base = "https://api.elections.kalshi.com/trade-api/v2/markets"
    counts, sample, quoted = {}, None, 0
    for series in KALSHI_PROP_SERIES:
        r = client.get_json(base, params={"series_ticker": series, "status": "open", "limit": 1000})
        if not r.ok:
            return "FAIL", f"{series}: {fail_detail(r)}", None
        markets = r.data.get("markets", [])
        counts[series] = len(markets)
        for m in markets:
            if float(m.get("yes_ask_dollars") or 0) > 0 and float(m.get("yes_bid_dollars") or 0) > 0:
                quoted += 1
                if sample is None or float(m.get("volume_fp") or 0) > float(sample.get("volume_fp") or 0):
                    sample = m
        if series == "KXNBAPTS":
            save_sample("kalshi_pts_markets", r.data)
    total = sum(counts.values())
    detail = f"{total} open NBA player-prop markets ({quoted} with a two-sided quote); " + \
             ", ".join(f"{k}={v}" for k, v in counts.items())
    if sample:
        detail += (f". Most traded: '{sample['title']}' yes bid/ask "
                   f"{sample['yes_bid_dollars']}/{sample['yes_ask_dollars']}, volume {sample['volume_fp']}")
    return ("PASS" if total else "WARN"), detail, {"markets_per_series": counts}


def odds_api() -> tuple[str, str, dict | None]:
    key = get_secret("ODDS_API_KEY")
    if not key:
        return "SKIP", "ODDS_API_KEY not set in .env — get a free key, then re-run", None
    client = make_client("odds_api")
    base = "https://api.the-odds-api.com/v4"
    cfg = CONFIG["odds"]["the_odds_api"]
    # /events is free (0 credits) and lists upcoming + in-play games.
    r = client.get_json(f"{base}/sports/{cfg['sport_key']}/events", params={"apiKey": key})
    if not r.ok:
        return "FAIL", fail_detail(r), None
    events = r.data
    if not events:
        return "WARN", "key works but no NBA events listed", None
    live = [e for e in events if seconds_since(e["commence_time"]) > 0]
    event = (live or events)[0]
    lines, books_by_region, quota = [], {}, (None, None, None)
    # 1 market x 1 region per call = at most 1 credit each.
    for region in ("us", "us_dfs"):
        r = client.get_json(f"{base}/sports/{cfg['sport_key']}/events/{event['id']}/odds",
                            params={"apiKey": key, "regions": region, "markets": "player_points",
                                    "oddsFormat": cfg["odds_format"], "includeMultipliers": "true"})
        if not r.ok:
            lines.append(f"{region}: {fail_detail(r)}")
            continue
        save_sample(f"odds_api_player_points_{region}", r.data)
        books = r.data.get("bookmakers", [])
        books_by_region[region] = [b["key"] for b in books]
        n_out = sum(len(m["outcomes"]) for b in books for m in b["markets"])
        lines.append(f"{region}: {len(books)} books ({', '.join(books_by_region[region]) or 'none'}), "
                     f"{n_out} outcomes")
        quota = (r.headers.get("x-requests-used"), r.headers.get("x-requests-remaining"),
                 r.headers.get("x-requests-last"))
    state = "IN-PLAY" if event in live else "pre-game"
    detail = (f"{event['away_team']} @ {event['home_team']} ({state}) player_points -> " + " | ".join(lines)
              + f". Credits used/remaining/last call: {'/'.join(map(str, quota))}")
    return "PASS", detail, {"books_by_region": books_by_region, "event_state": state}


# --------------------------------------------------------------------------- main

COSTS_CREDITS = {"the_odds_api_player_props"}

TESTS = [  # (section, test name, source, function)
    ("NBA live feeds (cdn.nba.com)", "nba_live_scoreboard_raw", "cdn.nba.com", nba_live_raw),
    ("NBA live feeds (cdn.nba.com)", "nba_api_live_scoreboard", "cdn.nba.com", nba_api_live("scoreboard")),
    ("NBA live feeds (cdn.nba.com)", "nba_api_live_boxscore", "cdn.nba.com", nba_api_live("boxscore")),
    ("NBA live feeds (cdn.nba.com)", "nba_api_live_playbyplay", "cdn.nba.com", nba_api_live("playbyplay")),
    ("NBA stats (stats.nba.com)", "nba_api_matchups_v3", "stats.nba.com", nba_stats("matchups")),
    ("NBA stats (stats.nba.com)", "nba_api_league_gamelog", "stats.nba.com", nba_stats("gamelog")),
    ("ESPN (live feed)", "espn_scoreboard", "espn", espn_scoreboard),
    ("ESPN (live feed)", "espn_summary_final", "espn", espn_summary("final")),
    ("ESPN (live feed)", "espn_summary_live", "espn", espn_summary("live")),
    ("ESPN (live feed)", "espn_injuries", "espn", espn_injuries),
    ("Injuries (official)", "nba_official_injury_report", "official.nba.com", nba_injury_report),
    ("Odds", "kalshi_player_props", "kalshi", kalshi_props),
    ("Odds", "the_odds_api_player_props", "the-odds-api", odds_api),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", metavar="WORD",
                        help="run only tests whose name or source contains one of these words "
                             "(e.g. --only odds_api, --only espn kalshi)")
    args = parser.parse_args()
    if args.only:
        selected = [t for t in TESTS if any(w in t[1] or w in t[2] for w in args.only)]
    else:  # tests that spend paid/limited credits only run when asked for by name
        selected = [t for t in TESTS if t[1] not in COSTS_CREDITS]
        print(f"(skipping {', '.join(COSTS_CREDITS)}: spends Odds API credits — run with --only odds_api)")
    if not selected:
        parser.error(f"no test matches {args.only}; names: {', '.join(t[1] for t in TESTS)}")

    print(f"Smoke test @ {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    section = None
    for sec, name, source, fn in selected:
        if sec != section:
            print(f"\n{sec}")
            section = sec
        run(name, source, fn)

    # Merge into the previous results file so a partial run (--only) doesn't erase other tests.
    previous = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    merged = {r["test"]: r for r in previous.get("results", [])}
    merged.update({r["test"]: r for r in results})
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(
        {"context": {**previous.get("context", {}), **context}, "results": list(merged.values())}, indent=2))
    n = {s: sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "WARN", "SKIP")}
    print(f"\n{n['PASS']} pass, {n['FAIL']} fail, {n['WARN']} warn, {n['SKIP']} skip. "
          f"Details: {RESULTS_PATH.relative_to(ROOT)}; raw samples: {SAMPLES.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
