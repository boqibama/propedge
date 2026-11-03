"""`python -m propedge watch`: a live, READ-ONLY view of tonight's run in the Terminal.

Opens the database read-only (SQLite `mode=ro`; if that isn't possible, `PRAGMA query_only`), runs a
few short queries per refresh and holds no transaction between refreshes, so the poller is never
blocked. It reads the log file but never writes to it.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from propedge.config import ROOT, resolve_path
from propedge.db import models as m
from propedge.db.store import utc
from propedge.ids import EASTERN

RULE_NAME = {"rule_a": "A", "rule_b": "B"}


def readonly_engine(cfg: dict) -> Engine:
    url = cfg["paths"]["database_url"]
    if not url.startswith("sqlite:///"):
        raise SystemExit("watch reads the SQLite database only")
    path = (ROOT / url.removeprefix("sqlite:///")).resolve()
    if not path.exists():
        raise SystemExit(f"no database at {path}")

    def connect():
        try:            # true read-only: SQLite refuses any write, and closing never checkpoints the WAL
            return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5, check_same_thread=False)
        except sqlite3.OperationalError:   # e.g. WAL files missing while no poller runs
            con = sqlite3.connect(str(path), timeout=5, check_same_thread=False)
            con.execute("PRAGMA query_only=ON")
            return con

    engine = create_engine("sqlite://", creator=connect)

    @event.listens_for(engine, "connect")
    def _ro(conn, _record):
        conn.execute("PRAGMA query_only=ON")
    return engine


def default_game_day(session: Session, now: datetime) -> date:
    """The night being played: a game in progress decides; else today (US-Eastern), or yesterday before 6 am."""
    live = session.scalar(select(func.max(m.Game.game_date)).where(m.Game.status == "in"))
    if live:
        return live
    et = now.astimezone(EASTERN)
    return (et - timedelta(days=1)).date() if et.hour < 6 else et.date()


def ago(t: datetime | None, now: datetime) -> str:
    if t is None:
        return "never"
    s = (now - t).total_seconds()
    return f"{s:.0f}s ago" if s < 120 else f"{s / 60:.0f}m ago" if s < 7200 else f"{s / 3600:.1f}h ago"


def local(t: datetime | None) -> str:
    return "-" if t is None else t.astimezone().strftime("%H:%M:%S")


def clock(period: int | None, clock_s: float | None) -> str:
    if period is None:
        return ""
    q = f"Q{period}" if period <= 4 else f"OT{period - 4}"
    c = clock_s or 0
    return f"{q} {int(c // 60):02d}:{int(c % 60):02d}"


def log_problems(cfg: dict, since: datetime, limit: int = 6) -> tuple[list[str], int]:
    """WARNING/ERROR lines in the log since `since` (local time stamps): the last few, and how many."""
    path = resolve_path(cfg["paths"]["log_file"])
    if not path.exists():
        return [], 0
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (WARNING|ERROR|CRITICAL)\s")
    start = since.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    hits: deque[str] = deque(maxlen=limit)
    n = 0
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - 4_000_000))    # last ~4 MB is plenty for one night
        for raw in f:
            line = raw.decode("utf-8", "replace").rstrip()
            mt = pat.match(line)
            if mt and mt.group(1) >= start:
                n += 1
                hits.append(line)
    return list(hits), n


def poller_pids() -> list[str]:
    try:
        out = subprocess.run(["pgrep", "-f", "propedge (tonight|poll)"], capture_output=True, text=True, timeout=2)
        return out.stdout.split()
    except Exception:
        return []


def snapshot_gaps(session: Session, game_ids: list[str], gap_s: float) -> list[tuple[datetime, float]]:
    """Gaps in our ESPN polling while games were live (all games share one poller)."""
    ts = sorted({utc(t) for t in session.scalars(select(m.LiveGameSnapshot.fetched_at).where(
        m.LiveGameSnapshot.game_id.in_(game_ids), m.LiveGameSnapshot.status == "in",
        m.LiveGameSnapshot.is_mock.is_(False)))})
    return [(b, (b - a).total_seconds()) for a, b in zip(ts, ts[1:]) if (b - a).total_seconds() > gap_s]


def render(session: Session, cfg: dict, now: datetime, day: date | None = None) -> str:
    from propedge.edge.event_trades import frozen_trades, load_games
    from propedge.models.trade_sim import Fees

    day = day or default_game_day(session, now)
    games = session.scalars(select(m.Game).where(m.Game.game_date == day).order_by(m.Game.start_time)).all()
    gids = [g.id for g in games]
    out = [f"propedge watch · night of {day} · {now.astimezone().strftime('%H:%M:%S %Z')} · refresh "
           f"every few seconds · READ-ONLY (Ctrl-C to quit)", ""]

    # ------------------------------------------------------------------ poller health
    last_espn = session.scalar(select(func.max(m.LiveGameSnapshot.fetched_at)).where(
        m.LiveGameSnapshot.game_id.in_(gids), m.LiveGameSnapshot.is_mock.is_(False)))
    last_kalshi = session.scalar(select(func.max(m.OddsSnapshot.last_seen_at)).where(
        m.OddsSnapshot.ticker.in_(select(m.PropMarket.ticker).where(m.PropMarket.game_id.in_(gids)))))
    last_espn, last_kalshi = (utc(last_espn) if last_espn else None), (utc(last_kalshi) if last_kalshi else None)
    pids = poller_pids()
    live_n = sum(g.status == "in" for g in games)
    out.append(f"POLLER  {'running (pid ' + ', '.join(pids) + ')' if pids else 'NOT RUNNING'}   "
               f"last ESPN snapshot {local(last_espn)} ({ago(last_espn, now)})   "
               f"last Kalshi quote {local(last_kalshi)} ({ago(last_kalshi, now)})")
    warn = []
    if live_n and last_espn and (now - last_espn).total_seconds() > 60:
        warn.append(f"no ESPN snapshot for {ago(last_espn, now)} while {live_n} game(s) are live")
    if live_n and last_kalshi and (now - last_kalshi).total_seconds() > 30:
        warn.append(f"no Kalshi quote for {ago(last_kalshi, now)} while games are live")
    if live_n and not pids:
        warn.append("games are live but no poller process is running")
    for w in warn:
        out.append(f"  !! {w}")
    out.append("")

    # ------------------------------------------------------------------ games
    out.append("GAMES                 score      status        data behind (newest play / delivery)   last fetch")
    for g in games:
        snap = session.scalars(select(m.LiveGameSnapshot).where(m.LiveGameSnapshot.game_id == g.id,
                                                                m.LiveGameSnapshot.is_mock.is_(False))
                               .order_by(m.LiveGameSnapshot.fetched_at.desc()).limit(1)).first()
        name = f"{g.away} @ {g.home}"
        score = f"{g.away_score}-{g.home_score}" if g.away_score is not None else ""
        if g.status == "pre":
            out.append(f"  {name:<19} {'':<10} tip {local(utc(g.start_time)) if g.start_time else '?'}")
            continue
        if g.status == "post":
            out.append(f"  {name:<19} {score:<10} final")
            continue
        if snap is None:
            out.append(f"  {name:<19} {score:<10} live (no snapshot yet)")
            continue
        behind = f"{snap.feed_lag_s:.0f}s" if snap.feed_lag_s is not None else "-"
        delay = f"{snap.play_delay_s:.0f}s" if snap.play_delay_s is not None else "-"
        flags = (" clock stopped" if snap.clock_running is False else "") + (" STALE" if snap.is_stale else "")
        out.append(f"  {name:<19} {snap.away_score}-{snap.home_score:<7} {clock(snap.period, snap.clock_s):<13} "
                   f"{behind:>5} / {delay:<5}{flags:<20}          {ago(utc(snap.fetched_at), now)}")
    out.append("")

    # ------------------------------------------------------------------ gaps and errors
    gaps = snapshot_gaps(session, gids, cfg["live_feed"]["gap_warn_s"])
    first = min((utc(g.start_time) for g in games if g.start_time), default=now) - timedelta(hours=2)
    problems, n_problems = log_problems(cfg, first)
    out.append(f"GAPS & ERRORS  ESPN polling gaps > {cfg['live_feed']['gap_warn_s']}s while live: {len(gaps)}   "
               f"log warnings/errors tonight: {n_problems}")
    for t, s in gaps[-3:]:
        out.append(f"  gap of {s:.0f}s ending {local(t)}")
    for line in problems:
        out.append(f"  {line[11:19]} {line[20:][:150]}")
    out.append("")

    # ------------------------------------------------------------------ events and paper trades
    fr = cfg["frozen_rules"]
    counted = day >= date.fromisoformat(fr["judge_from"])
    loaded = list(load_games(session, [g.id for g in games if g.status != "pre"], fr["blowout_lead"],
                             fr["blowout_from_min"], fr["foul_n"], only_event_players=True))
    events = sorted((e for g in loaded for e in g.events), key=lambda e: e.t)
    trades = frozen_trades(loaded, cfg, Fees.from_config(cfg), now=now)
    names = dict(session.execute(select(m.Player.id, m.Player.name)).all())
    out.append(f"EVENTS TONIGHT  {sum(e.type == 'foul4' for e in events)} fourth fouls, "
               f"{sum(e.type == 'blowout' for e in events)} blowouts   ·   Rules A/B paper trades: entry "
               f"{fr['entry_delay_s']}s after the event at the ask, exit at the bid, fees included   ·   "
               + ("COUNTS for the frozen test" if counted else f"pre-freeze night: NOT counted (from {fr['judge_from']})"))
    if not events:
        out.append("  none yet")
    by_event = {k: d for k, d in trades.groupby("event")} if not trades.empty else {}
    for e in events[-25:]:
        g = e.game_id.split("_")
        who = names.get(e.fouler, f"#{e.fouler}") if e.type == "foul4" else f"lead {e.lead}"
        what = "4th foul" if e.type == "foul4" else "BLOWOUT"
        d = by_event.get(e.key)
        if d is None:
            rule = "no rule (4th quarter or later)" if e.type == "foul4" else "no rule"
            out.append(f"  {local(e.t)}  {g[1]}@{g[2]:<4} {what:<8} {who:<24} {clock(e.period, None)[:3]:<4} {rule}")
            continue
        r = RULE_NAME[d["rule"].iloc[0]]
        head = f"  {local(e.t)}  {g[1]}@{g[2]:<4} {what:<8} {who:<24} {clock(e.period, None)[:3]:<4} Rule {r}: "
        out.append(head + _trade_text(d, names, now, e.type == "blowout"))
    if len(events) > 25:
        out.append(f"  … {len(events) - 25} earlier events not shown")
    if not trades.empty and "pnl" in trades:
        tot = []
        for rule, d in trades.groupby("rule"):
            done = d[(d["status"] == "closed") & d["pnl"].notna()]
            ev_pnl = done.groupby("event")["pnl"].mean()
            tot.append(f"Rule {RULE_NAME[rule]}: {len(ev_pnl)} closed events, avg {_c(ev_pnl.mean())}/contract"
                       if len(ev_pnl) else f"Rule {RULE_NAME[rule]}: none closed yet")
        out.append("  tonight so far → " + "   ".join(tot))
    return "\n".join(out)


def _c(x) -> str:
    return "-" if x is None or x != x else f"{x * 100:+.1f}¢"


def _trade_text(d: pd.DataFrame, names: dict, now: datetime, many: bool) -> str:
    st = d["status"].iloc[0]
    if st == "waiting":
        secs = (d["t_entry"].iloc[0] - now).total_seconds()
        return f"buying in {secs:.0f}s"
    traded = d[d["pnl"].notna()] if "pnl" in d else d.iloc[0:0]
    if many:
        if traded.empty:
            return f"{len(d)} starters, none tradable ({', '.join(d['reason'].dropna().unique())})"
        left = (d["t_exit"].iloc[0] - now).total_seconds()
        tag = "OPEN, marked at bid now" if st == "open" else "closed"
        return (f"{len(traded)}/{len(d)} starters traded, avg {_c(traded['pnl'].mean())}/contract "
                f"[{tag}{f', sells in {left:.0f}s' if st == 'open' else ''}]")
    row = d.iloc[0]
    if pd.isna(row.get("pnl")):
        return f"no trade ({row.get('reason')})"
    left = (row["t_exit"] - now).total_seconds()
    tag = f"OPEN, sells in {left:.0f}s, marked at bid now" if st == "open" else f"closed ({row['how']})"
    return (f"NO on {row['line'] + 0.5:.0f}+ bought {row['entry']:.2f}, {'now' if st == 'open' else 'sold'} "
            f"{row['exit_price']:.2f} → {_c(row['pnl'])}/contract [{tag}]")


def run(cfg: dict, every_s: float = 5.0, day: date | None = None, once: bool = False) -> None:
    engine = readonly_engine(cfg)
    while True:
        t0 = time.monotonic()
        now = datetime.now(timezone.utc)
        try:
            with Session(engine) as session:       # one short read per refresh; closed before sleeping
                text = render(session, cfg, now, day)
        except Exception as e:                     # never die mid-night: show the problem and retry
            text = f"propedge watch · {now.astimezone():%H:%M:%S} · read failed, retrying: {e!r}"
        if once:
            print(text)
            return
        print("\033[H\033[2J" + text + f"\n\n(rendered in {time.monotonic() - t0:.1f}s)", flush=True)
        time.sleep(max(0.5, every_s - (time.monotonic() - t0)))
