"""Edge test: how fast does Kalshi reprice after game events we'd also see — ~50 s later?

Events (true time = ESPN's wall-clock time of the play):
  foul4 / foul5  a player's 4th / 5th personal foul        -> his "over" prices should drop
  blowout        the lead first reaches `blowout_lead` in the second half -> starters' overs drop
  exit           a starter checks out before the 4th quarter and never returns (not a blowout) -> overs drop
Market reaction comes from Kalshi's public trade history (exact trade timestamps) for the player's
contracts. For each event with a clear move we measure what share of the eventual move had happened
at each delay after the event. If most of it is done by +50 s, there's no edge left for us there.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.backtest.live import load_game
from propedge.db import models as m
from propedge.db.store import PlayerResolver, utc
from propedge.http import HttpClient
from propedge.ids import parse_kalshi_event
from propedge.models.game_state import elapsed_minutes, is_personal_foul, replay_states
from propedge.odds.kalshi import parse_market

log = logging.getLogger(__name__)
OFFSETS = [-60, -20, 0, 10, 20, 30, 45, 50, 60, 90, 120, 180, 300]


# ----------------------------------------------------------------------------- Kalshi history

def all_markets(client: HttpClient, base: str, series: str) -> list:
    out = []
    for status in ("settled", "closed", "open"):
        cursor = None
        while True:
            params = {"series_ticker": series, "status": status, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            r = client.get_json(f"{base}/markets", params=params)
            if not r.ok:
                break
            out += [q for q in (parse_market(mk, "pts", r.fetched_at) for mk in r.data.get("markets", [])) if q]
            cursor = r.data.get("cursor")
            if not cursor:
                break
    return out


def trades(client: HttpClient, base: str, ticker: str, start: datetime, end: datetime, cache_dir: Path) -> pd.DataFrame:
    """Kalshi trades for one contract in [start, end] (cached on disk: history doesn't change)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{ticker}_{int(start.timestamp())}_{int(end.timestamp())}.json"
    if f.exists():
        rows = json.loads(f.read_text())
    else:
        rows, cursor, ok = [], None, True
        while True:
            params = {"ticker": ticker, "min_ts": int(start.timestamp()), "max_ts": int(end.timestamp()), "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            r = client.get_json(f"{base}/markets/trades", params=params)
            if not r.ok:
                log.warning("trades %s: %s", ticker, r.error)
                ok = False
                break
            rows += [{"t": t["created_time"], "p": float(t["yes_price_dollars"]), "n": float(t["count_fp"])}
                     for t in r.data.get("trades", [])]
            cursor = r.data.get("cursor")
            if not cursor:
                break
        if ok:            # never cache a failed fetch as "no trades"
            f.write_text(json.dumps(rows))
    df = pd.DataFrame(rows, columns=["t", "p", "n"])
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"], utc=True)
        df = df.sort_values("t")
    return df


def candle_mids(client: HttpClient, base: str, series: str, ticker: str, start: datetime, end: datetime,
                cache_dir: Path) -> pd.DataFrame:
    """Kalshi 1-minute candles -> mid of the closing bid/ask at the end of each minute."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"candles_{ticker}_{int(start.timestamp())}_{int(end.timestamp())}.json"
    if f.exists():
        rows = json.loads(f.read_text())
    else:
        r = client.get_json(f"{base}/series/{series}/markets/{ticker}/candlesticks",
                            params={"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()),
                                    "period_interval": 1})
        if not r.ok:
            return pd.DataFrame(columns=["t", "p"])
        rows = []
        for c in r.data.get("candlesticks", []):
            bid, ask = float(c["yes_bid"]["close_dollars"]), float(c["yes_ask"]["close_dollars"])
            if 0 < bid and 0 < ask < 1:
                rows.append({"t": datetime.fromtimestamp(c["end_period_ts"], start.tzinfo).isoformat(),
                             "p": (bid + ask) / 2})
        f.write_text(json.dumps(rows))
    df = pd.DataFrame(rows, columns=["t", "p"])
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


def quote_mids(session: Session, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Our own stored quotes (every ~3 s during live games) -> mid at each change."""
    rows = session.execute(select(m.OddsSnapshot.first_seen_at, m.OddsSnapshot.yes_bid, m.OddsSnapshot.yes_ask)
                           .where(m.OddsSnapshot.ticker == ticker, m.OddsSnapshot.first_seen_at >= start,
                                  m.OddsSnapshot.first_seen_at <= end)
                           .order_by(m.OddsSnapshot.first_seen_at)).all()
    df = pd.DataFrame([{"t": utc(t), "p": (b + a) / 2} for t, b, a in rows if b and a], columns=["t", "p"])
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


# ----------------------------------------------------------------------------- events

def find_events(g, blowout_lead: float, exit_before_min: float, exit_max_minutes: float) -> list[dict]:
    """Foul 4/5, blowout onset and starter exits in one game (players keyed by our player id)."""
    events, fouls = [], {}
    for p in sorted(g.plays, key=lambda p: p.sequence):
        if p.wallclock is None:
            continue
        if is_personal_foul(p.play_type) and p.provider_player_ids:
            pid = p.provider_player_ids[0]
            fouls[pid] = fouls.get(pid, 0) + 1
            if fouls[pid] in (4, 5):
                events.append({"kind": f"foul{fouls[pid]}", "players": [pid], "time": p.wallclock,
                               "elapsed": elapsed_minutes(p.period, p.clock_s)})
    # blowout onset (second half)
    for p in sorted(g.plays, key=lambda p: p.sequence):
        e = elapsed_minutes(p.period, p.clock_s)
        if p.wallclock and e >= 24 and p.home_score is not None and abs(p.home_score - p.away_score) >= blowout_lead:
            starters = [pid for ids in g.starters.values() for pid in ids]
            events.append({"kind": "blowout", "players": starters, "time": p.wallclock, "elapsed": e})
            break
    # starter exits: last check-out before `exit_before_min`, never back, few minutes, not a blowout
    last_out = {}
    for p in sorted(g.plays, key=lambda p: p.sequence):
        if p.play_type == "Substitution" and len(p.provider_player_ids) == 2 and p.wallclock:
            last_out[p.provider_player_ids[1]] = (p.wallclock, elapsed_minutes(p.period, p.clock_s),
                                                  abs((p.home_score or 0) - (p.away_score or 0)))
    came_back = set()
    for p in g.plays:
        if p.play_type == "Substitution" and len(p.provider_player_ids) == 2:
            pin = p.provider_player_ids[0]
            if pin in last_out and elapsed_minutes(p.period, p.clock_s) > last_out[pin][1]:
                came_back.add(pin)
    for pid, (t, e, lead) in last_out.items():
        is_starter = any(pid in ids for ids in g.starters.values())
        if is_starter and pid not in came_back and e < exit_before_min and lead < blowout_lead \
                and g.final_min.get(pid, 99) <= exit_max_minutes:
            events.append({"kind": "exit", "players": [pid], "time": t, "elapsed": e})
    return events


def reaction(tr: pd.DataFrame, t0: datetime) -> dict | None:
    """Share of the eventual price move done at each offset (seconds) after t0, from trades."""
    if tr.empty:
        return None
    before = tr[(tr["t"] >= t0 - timedelta(seconds=240)) & (tr["t"] < t0 - timedelta(seconds=60))]
    after = tr[(tr["t"] >= t0 + timedelta(seconds=300)) & (tr["t"] <= t0 + timedelta(seconds=900))]
    if len(before) < 1 or len(after) < 1:
        return None
    pre, post = float(before["p"].iloc[-1]), float(after["p"].median())
    move = post - pre
    out = {"pre": pre, "post": post, "move": move, "clear": abs(move) >= 0.05}
    for off in OFFSETS:
        upto = tr[tr["t"] <= t0 + timedelta(seconds=off)]
        price = float(upto["p"].iloc[-1]) if len(upto) else pre
        out[off] = float(np.clip((price - pre) / move, -1, 2)) if out["clear"] else float("nan")
    return out


def run(session: Session, cfg: dict, client: HttpClient, cache_dir: Path, source: str = "trades") -> pd.DataFrame:
    """source: "trades" (Kalshi trade history), "candles" (1-minute bid/ask mids) or "quotes" (our
    own ~3 s quote history — only for games we polled live)."""
    base, series = cfg["odds"]["kalshi"]["base_url"], cfg["odds"]["kalshi"]["series"]["pts"]
    rp = cfg["repricing"]
    markets = all_markets(client, base, series)
    resolver = PlayerResolver(session)
    by_game: dict[str, list] = {}
    for q in markets:
        parsed = parse_kalshi_event(q.event_ticker)
        if not parsed:
            continue
        d, away, home = parsed
        pid = resolver.from_name(q.player_name, q.team)
        if pid is not None:
            by_game.setdefault(f"{d.isoformat()}_{away}_{home}", []).append((str(pid), q))
    rows = []
    for gid, contracts in sorted(by_game.items()):
        g = load_game(session, gid)
        if g is None:
            continue
        events = find_events(g, rp["blowout_lead"], rp["exit_before_min"], rp["exit_max_minutes"])
        # Placebo: the same contracts 15 minutes before each event, if that moment is itself event-free.
        times = [e["time"] for e in events]
        placebos = [{**e, "kind": "placebo", "time": e["time"] - timedelta(minutes=15)} for e in events
                    if not any(abs((e["time"] - timedelta(minutes=15) - t).total_seconds()) < 300 for t in times)]
        for ev in events + placebos:
            t0 = ev["time"]
            for pid in ev["players"]:
                mine = [q for p, q in contracts if p == pid]
                best = None
                for q in mine:
                    lo, hi = t0 - timedelta(minutes=10), t0 + timedelta(minutes=15)
                    if source == "trades":
                        tr = trades(client, base, q.ticker, lo, hi, cache_dir)
                    elif source == "candles":
                        tr = candle_mids(client, base, series, q.ticker, lo, hi, cache_dir)
                    else:
                        tr = quote_mids(session, q.ticker, lo, hi)
                    r = reaction(tr, t0)
                    if r and (best is None or abs(r["move"]) > abs(best[1]["move"])):     # the contract that moved most
                        best = (q, r, len(tr))
                if best:
                    q, r, n = best
                    rows.append({"game_id": gid, "kind": ev["kind"], "player_id": pid, "ticker": q.ticker,
                                 "line": q.line, "elapsed": ev["elapsed"], "event_time": t0, "trades": n, **r})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, feed_delay_s: float, n_candidates: dict | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = []
    events = df[df["kind"] != "placebo"]
    for kind, d in [("ALL events", events)] + list(df.groupby("kind")):
        clear = d[d["clear"]]
        row = {"event": kind, "measured": len(d), "share_with_clear_move": float(d["clear"].mean()),
               "n_clear": len(clear), "median_move": float(clear["move"].abs().median()) if len(clear) else float("nan")}
        for off in OFFSETS:
            row[f"{off:+d}s"] = float(clear[off].median()) if len(clear) else float("nan")
        row[f"done_by_{feed_delay_s:.0f}s"] = float(clear[50].median()) if len(clear) else float("nan")
        out.append(row)
    return pd.DataFrame(out)
