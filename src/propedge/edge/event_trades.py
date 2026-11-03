"""Events as trades: detect 4th fouls and blowout onsets from the snapshots we stored, then simulate
buying a contract shortly after our data showed the event and selling it a few minutes later.

Events are found from our own ESPN snapshots, independently of the model, so every event counts — not
only those where the model happened to log a candidate. Each event is one independent unit: a blowout
moves a whole team's contracts together, so its trades are averaged into one number before anything
is summed across events.

Two uses:
  exploratory  (`event-trades`)  entry delays × holds × exits × three ways of choosing the contract
  frozen test  (`rules-report`)  Rules A and B exactly as written in DECISIONS #62, nights from
                                 `frozen_rules.judge_from` only
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.db import models as m
from propedge.db.store import utc
from propedge.models.game_state import elapsed_minutes
from propedge.models.trade_sim import Fees, Quote, QuoteSeries, bootstrap_ci, closest_to_half, round_trip, yes_mid

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoxLine:
    team: str
    starter: bool
    pf: int
    pts: int


@dataclass(frozen=True)
class Frame:
    """One stored ESPN snapshot as our data showed it."""
    t: datetime
    status: str
    period: int | None
    clock_s: float | None
    home: int
    away: int
    box: dict[int, BoxLine] | None       # None = box score unchanged since the previous frame


@dataclass
class Event:
    type: str                            # foul4 | blowout
    game_id: str
    t: datetime                          # fetch time of the first snapshot that showed it
    period: int
    elapsed: float
    players: dict[int, str]              # player id -> role (fouler | starter)
    pts: dict[int, int] = field(default_factory=dict)    # their points at the event
    fouler: int | None = None
    lead: int | None = None

    @property
    def key(self) -> str:
        who = self.fouler if self.type == "foul4" else "game"
        return f"{self.game_id}|{self.type}|{who}|{self.t.isoformat(timespec='seconds')}"

    @property
    def game_date(self) -> date:
        return date.fromisoformat(self.game_id[:10])


def detect_events(game_id: str, frames: Iterable[Frame], blowout_lead: int = 20, blowout_from_min: float = 24,
                  foul_n: int = 4) -> list[Event]:
    """4th fouls and blowout onsets as our data showed them (DECISIONS #62).
      foul4    a player's fouls go from < foul_n to ≥ foul_n, the first time in the game (ESPN sometimes takes a
               foul back, 4 -> 3 -> 4; that's still one 4th foul)            -> him ("fouler")
      blowout  the lead goes from < blowout_lead to ≥ it at game minute ≥ blowout_from_min, the first time,
               and only if we hadn't already seen such a lead (e.g. polling started mid-blowout)
               -> both teams' ESPN starters who played ("starter")
    The first in-game frame only records the state: a restart mid-game isn't an event."""
    events: list[Event] = []
    prev_pf: dict[int, int] = {}
    fouled_out_n: set[int] = set()       # players whose foul_n-th foul already counted
    box: dict[int, BoxLine] = {}
    prev_lead: int | None = None
    big_lead_seen = False
    for f in frames:
        if f.status != "in" or f.period is None:
            continue
        first = prev_lead is None
        elapsed = elapsed_minutes(f.period, f.clock_s)
        if f.box is not None:
            box = f.box
            for pid, b in box.items():
                prev = prev_pf.get(pid)
                prev_pf[pid] = b.pf
                if b.pf >= foul_n and pid not in fouled_out_n:
                    fouled_out_n.add(pid)
                    if first or prev is None:
                        continue          # already there when we first saw him: not an event
                    events.append(Event("foul4", game_id, f.t, f.period, elapsed, {pid: "fouler"}, {pid: b.pts},
                                        fouler=pid))
        lead = abs(f.home - f.away)
        second_half = elapsed >= blowout_from_min
        if (not first and second_half and lead >= blowout_lead and prev_lead < blowout_lead
                and not big_lead_seen):
            starters = {pid: "starter" for pid, b in box.items() if b.starter}
            events.append(Event("blowout", game_id, f.t, f.period, elapsed, starters,
                                {pid: box[pid].pts for pid in starters}, lead=lead))
        if second_half and lead >= blowout_lead:
            big_lead_seen = True
        prev_lead = lead
    return events


# ----------------------------------------------------------------------------- loading from the database

def frames_from_db(session: Session, game_id: str) -> list[Frame]:
    snaps = session.execute(select(m.LiveGameSnapshot.id, m.LiveGameSnapshot.fetched_at, m.LiveGameSnapshot.status,
                                   m.LiveGameSnapshot.period, m.LiveGameSnapshot.clock_s, m.LiveGameSnapshot.home_score,
                                   m.LiveGameSnapshot.away_score)
                            .where(m.LiveGameSnapshot.game_id == game_id, m.LiveGameSnapshot.provider == "espn",
                                   m.LiveGameSnapshot.is_mock.is_(False))
                            .order_by(m.LiveGameSnapshot.fetched_at)).all()
    if not snaps:
        return []
    rows = session.execute(select(m.LivePlayerSnapshot.snapshot_id, m.LivePlayerSnapshot.player_id,
                                  m.LivePlayerSnapshot.team, m.LivePlayerSnapshot.starter,
                                  m.LivePlayerSnapshot.did_not_play, m.LivePlayerSnapshot.pf, m.LivePlayerSnapshot.pts)
                           .where(m.LivePlayerSnapshot.snapshot_id.in_([s.id for s in snaps]))).all()
    boxes: dict[int, dict[int, BoxLine]] = {}
    for sid, pid, team, starter, dnp, pf, pts in rows:
        if dnp:
            continue
        boxes.setdefault(sid, {})[pid] = BoxLine(team, bool(starter), int(pf or 0), int(pts or 0))
    return [Frame(utc(s.fetched_at), s.status, s.period, s.clock_s, s.home_score or 0, s.away_score or 0,
                  boxes.get(s.id)) for s in snaps]


class PointsTimeline:
    """A player's points as our box score showed them over time (to know when a rung was decided)."""

    def __init__(self, frames: list[Frame]):
        self.t: list[datetime] = []
        self.pts: list[dict[int, int]] = []
        for f in frames:
            if f.box is not None:
                self.t.append(f.t)
                self.pts.append({pid: b.pts for pid, b in f.box.items()})

    def at(self, pid: int, t: datetime) -> int | None:
        import bisect
        i = bisect.bisect_right(self.t, t) - 1
        return None if i < 0 else self.pts[i].get(pid)


@dataclass
class GameData:
    game_id: str
    events: list[Event]
    timeline: PointsTimeline
    markets: pd.DataFrame                    # ticker, player_id, line, result
    series: dict[str, QuoteSeries]


def load_games(session: Session, game_ids: list[str], blowout_lead: int, blowout_from_min: float,
               foul_n: int, only_event_players: bool = False) -> Iterator[GameData]:
    """`only_event_players`: load Kalshi quotes only for players in an event (enough for the rules; `watch`)."""
    for gid in game_ids:
        frames = frames_from_db(session, gid)
        if not frames:
            continue
        events = detect_events(gid, frames, blowout_lead, blowout_from_min, foul_n)
        q = select(m.PropMarket.ticker, m.PropMarket.player_id, m.PropMarket.line, m.PropMarket.result).where(
            m.PropMarket.game_id == gid, m.PropMarket.stat == "pts", m.PropMarket.is_mock.is_(False))
        if only_event_players:
            q = q.where(m.PropMarket.player_id.in_({p for e in events for p in e.players}))
        mk = pd.DataFrame(session.execute(q).all(), columns=["ticker", "player_id", "line", "result"])
        series: dict[str, list[Quote]] = {}
        if len(mk):
            for t, a, b, bid, ask, bs, as_ in session.execute(
                    select(m.OddsSnapshot.ticker, m.OddsSnapshot.first_seen_at, m.OddsSnapshot.last_seen_at,
                           m.OddsSnapshot.yes_bid, m.OddsSnapshot.yes_ask, m.OddsSnapshot.yes_bid_size,
                           m.OddsSnapshot.yes_ask_size).where(m.OddsSnapshot.ticker.in_(mk["ticker"].tolist()),
                                                              m.OddsSnapshot.is_mock.is_(False))):
                series.setdefault(t, []).append(Quote(utc(a), utc(b), bid, ask, bs, as_))
        yield GameData(gid, events, PointsTimeline(frames), mk, {t: QuoteSeries(q) for t, q in series.items()})


def game_ids_for(session: Session, since: date | None, until: date | None) -> list[str]:
    q = select(m.Game.id).where(m.Game.id.in_(select(m.LiveGameSnapshot.game_id).distinct()))
    if since:
        q = q.where(m.Game.game_date >= since)
    if until:
        q = q.where(m.Game.game_date <= until)
    return sorted(session.scalars(q))


def model_picks(session: Session, game_ids: list[str]) -> dict[tuple[str, int, str], list[dict]]:
    """The engine's event candidates, keyed (game, player, event type); live ones win over replayed."""
    out: dict[tuple[str, int, str], list[dict]] = {}
    for f in session.scalars(select(m.Flag).where(m.Flag.kind == "event", m.Flag.game_id.in_(game_ids))):
        trig = (f.inputs or {}).get("trigger", {})
        out.setdefault((f.game_id, f.player_id, trig.get("type")), []).append(
            {"t": utc(f.created_at), "ticker": f.ticker, "side": f.side, "edge": f.edge, "mock": bool(f.is_mock)})
    return out


def model_pick_for(picks: dict, ev: Event, pid: int, tol_s: float = 2.0) -> tuple[str, str] | None:
    cands = [c for c in picks.get((ev.game_id, pid, ev.type), []) if abs((c["t"] - ev.t).total_seconds()) <= tol_s]
    if not cands:
        return None
    live = [c for c in cands if not c["mock"]]
    best = max(live or cands, key=lambda c: c["edge"])
    return best["ticker"], best["side"]


# ----------------------------------------------------------------------------- simulation

def _decider(g: GameData, pid: int, line: float, side: str):
    """Settled value of our side at t if the rung is already decided (the player passed the line)."""
    def decided(t: datetime) -> float | None:
        pts = g.timeline.at(pid, t)
        if pts is not None and pts > line:
            return 1.0 if side == "yes" else 0.0
        return None
    return decided


def rungs_at(g: GameData, pid: int, t: datetime, max_age_s: float) -> list[tuple[str, float, Quote | None]]:
    mk = g.markets[g.markets["player_id"] == pid]
    return [(r.ticker, float(r.line), g.series[r.ticker].at(t, max_age_s) if r.ticker in g.series else None)
            for r in mk.itertuples()]


def simulate(games: Iterable[GameData], cfg: dict, fees: Fees, picks: dict | None = None,
             roles: dict[str, str] | None = None) -> pd.DataFrame:
    """Exploratory grid: every event × player × selection × entry delay × hold × exit. One row per trade
    (or per missing trade, with `reason`)."""
    et = cfg["event_trades"]
    sides = roles or {"fouler": "no", "starter": "no"}
    lo, hi = et["all_rungs_mid_range"]
    rows = []
    for g in games:
        for ev in g.events:
            for pid, role in ev.players.items():
                side = sides.get(role)
                if side is None:
                    continue
                for delay in et["entry_delays_s"]:
                    t_in = ev.t + timedelta(seconds=delay)
                    rungs = rungs_at(g, pid, t_in, et["max_quote_age_s"])
                    if not rungs:
                        continue
                    chosen: dict[str, list[tuple[str, str]]] = {"closest50": [], "closest50_any": [], "all": [],
                                                                "model": []}
                    if c50 := closest_to_half(rungs, cfg["frozen_rules"]["rung_max_spread"]):
                        chosen["closest50"].append((c50, side))
                    if c50 := closest_to_half(rungs):
                        chosen["closest50_any"].append((c50, side))
                    for ticker, line, q in rungs:
                        mid = yes_mid(q)
                        if mid is not None and lo <= mid <= hi and line >= ev.pts.get(pid, 0):
                            chosen["all"].append((ticker, side))
                    if picks is not None and (mp := model_pick_for(picks, ev, pid)):
                        chosen["model"].append(mp)
                    for sel, items in chosen.items():
                        for ticker, s in items:
                            if ticker not in g.series:
                                continue
                            line = float(g.markets.loc[g.markets["ticker"] == ticker, "line"].iloc[0])
                            dec = _decider(g, pid, line, s)
                            for hold in et["hold_s"]:
                                for mode in ("bid", "limit"):
                                    tr = round_trip(g.series[ticker], s, t_in, t_in + timedelta(seconds=hold), fees,
                                                    et["max_quote_age_s"], mode, et["limit_wait_s"], dec)
                                    row = dict(event=ev.key, type=ev.type, role=role, game_id=ev.game_id,
                                               game_date=ev.game_date, period=ev.period, player_id=pid,
                                               selection=sel, delay_s=delay, hold_s=hold, exit=mode,
                                               ticker=ticker, line=line, side=s)
                                    if isinstance(tr, str):
                                        row["reason"] = tr
                                    else:
                                        row.update(entry=tr.entry, exit_price=tr.exit, gross=tr.gross, fees=tr.fees,
                                                   pnl=tr.pnl, how=tr.how, ask_size=tr.ask_size)
                                    rows.append(row)
    return pd.DataFrame(rows)


def by_event(trades: pd.DataFrame, group: list[str], level: float = 0.90, n_boot: int = 2000,
             seed: int = 7, order_contracts: float = 10) -> pd.DataFrame:
    """Collapse trades to one number per event (mean per-contract P&L), then summarise across events."""
    t = trades.dropna(subset=["pnl"]) if "pnl" in trades else trades.iloc[0:0]
    if t.empty:
        return pd.DataFrame()
    t = t.assign(limit_filled=(t["how"] == "limit").astype(float).where(t["exit"] == "limit"),
                 big_ask=(t["ask_size"] >= order_contracts).astype(float))
    ev = t.groupby(group + ["event"]).agg(pnl=("pnl", "mean"), gross=("gross", "mean"), fees=("fees", "mean"),
                                          trades=("pnl", "size"), pnl_sum=("pnl", "sum"), filled=("limit_filled", "mean"),
                                          big_ask=("big_ask", "mean")).reset_index()
    rows = []
    for key, d in ev.groupby(group):
        lo, hi = bootstrap_ci(d["pnl"], n_boot, level, seed)
        rows.append({**dict(zip(group, key if isinstance(key, tuple) else (key,))),
                     "events": len(d), "trades": int(d["trades"].sum()), "gross": d["gross"].mean(),
                     "fees": d["fees"].mean(), "net": d["pnl"].mean(), "total_per_contract": d["pnl_sum"].sum(), "ci_lo": lo, "ci_hi": hi,
                     "events_positive": float((d["pnl"] > 0).mean()), "limit_fill": d["filled"].mean(),
                     "ask_ge_order": d["big_ask"].mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- frozen test (DECISIONS #62)

def frozen_trades(games: Iterable[GameData], cfg: dict, fees: Fees, now: datetime | None = None) -> pd.DataFrame:
    """Rules A and B exactly as frozen: one row per trade (or missing trade, with `reason`).
    `now` (live view): trades still running are marked at the bid now (status "open"), and trades whose
    entry time hasn't come yet are listed as "waiting". Without `now` every trade is final ("closed")."""
    fr = cfg["frozen_rules"]
    rows = []
    for g in games:
        for ev in g.events:
            for name in ("rule_a", "rule_b"):
                r = fr[name]
                if ev.type != r["event"] or ("max_period" in r and ev.period > r["max_period"]):
                    continue
                t_in = ev.t + timedelta(seconds=fr["entry_delay_s"])
                t_out = t_in + timedelta(seconds=r["hold_s"])
                status = "closed" if now is None or now >= t_out else "open" if now >= t_in else "waiting"
                players = [p for p, role in ev.players.items() if role == r["players"]]
                if not players:
                    rows.append(dict(rule=name, event=ev.key, game_date=ev.game_date, period=ev.period,
                                     reason="no_players", status=status))
                for pid in players:
                    base = dict(rule=name, event=ev.key, game_id=ev.game_id, type=ev.type, t_event=ev.t,
                                t_entry=t_in, t_exit=t_out, game_date=ev.game_date, period=ev.period, player_id=pid,
                                status=status)
                    if status == "waiting":
                        rows.append(base)
                        continue
                    ticker = closest_to_half(rungs_at(g, pid, t_in, fr["max_quote_age_s"]), fr["rung_max_spread"])
                    if ticker is None:
                        rows.append({**base, "reason": "no_rung_with_tight_spread"})
                        continue
                    line = float(g.markets.loc[g.markets["ticker"] == ticker, "line"].iloc[0])
                    tr = round_trip(g.series[ticker], r["side"], t_in, t_out if status == "closed" else now, fees,
                                    fr["max_quote_age_s"], "bid", decided=_decider(g, pid, line, r["side"]))
                    base.update(ticker=ticker, line=line)
                    if isinstance(tr, str):
                        rows.append({**base, "reason": tr})
                    else:
                        rows.append({**base, "entry": tr.entry, "exit_price": tr.exit, "gross": tr.gross,
                                     "fees": tr.fees, "pnl": tr.pnl, "how": tr.how, "ask_size": tr.ask_size})
    return pd.DataFrame(rows)


def frozen_summary(trades: pd.DataFrame, cfg: dict, by_night: bool) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    fr, et = cfg["frozen_rules"], cfg["event_trades"]["bootstrap"]
    detected = trades.groupby(["rule"] + (["game_date"] if by_night else []))["event"].nunique().rename("detected")
    t = trades.assign(exit="bid")
    s = by_event(t, ["rule"] + (["game_date"] if by_night else []), et["level"], et["n"], et["seed"],
                 cfg["odds"]["kalshi"]["order_contracts"])
    if s.empty:
        return detected.reset_index()
    s = detected.reset_index().merge(s, how="left")
    s["pnl_10_contracts"] = s["total_per_contract"] * cfg["odds"]["kalshi"]["order_contracts"]
    return s.drop(columns=["limit_fill", "total_per_contract"])
