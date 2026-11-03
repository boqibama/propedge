"""Live edge finder (PAPER ONLY): price every Kalshi points rung of a game in progress with the live
engine, store the predictions, and log candidates with the live no-flag rules.

Used by the poller during games and by `replay-live` on recorded nights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.data.live_base import LiveGameState
from propedge.db import models as m
from propedge.db.store import PlayerResolver, utc
from propedge.edge.pregame import assign_statuses, model_version, run_pregame
from propedge.models.game_state import elapsed_minutes, replay_states
from propedge.models.kalshi_math import price_both_sides
from propedge.models.live_points import LiveInputs, simulate
from propedge.models.suppression import (QuotePoint, Reason, game_move_reason, market_reasons, player_move_reason,
                                         scoring_window_reason)
from propedge.odds.base import PropQuote

log = logging.getLogger(__name__)


@dataclass
class GameContext:
    game_id: str
    home: str
    pre: pd.DataFrame          # indexed by our player id: rate, min_mu, min_sd, pf_rate, confidence, low_reasons


def quotes_at(session: Session, game_id: str, at: datetime) -> list[PropQuote]:
    """The Kalshi quote current at `at` for every contract of a game (from our stored snapshots)."""
    markets = session.scalars(select(m.PropMarket).where(m.PropMarket.game_id == game_id)).all()
    out = []
    for mk in markets:
        snap = session.scalars(select(m.OddsSnapshot).where(m.OddsSnapshot.ticker == mk.ticker,
                                                            m.OddsSnapshot.first_seen_at <= at)
                               .order_by(m.OddsSnapshot.first_seen_at.desc()).limit(1)).first()
        if snap is None:
            continue
        d, away, home = mk.game_id.split("_")
        out.append(PropQuote(provider=mk.provider, ticker=mk.ticker, event_ticker=mk.event_ticker,
                             game_date=pd.Timestamp(d).date(), away=away, home=home, player_name=mk.player_name,
                             team=mk.team, stat=mk.stat, line=mk.line, yes_bid=snap.yes_bid, yes_ask=snap.yes_ask,
                             yes_bid_size=snap.yes_bid_size, yes_ask_size=snap.yes_ask_size, last_price=snap.last_price,
                             volume=snap.volume, open_interest=snap.open_interest, status=mk.status,
                             close_time=mk.close_time, fetched_at=utc(snap.last_seen_at)))
    return out


def foul_rates(session: Session, cfg: dict, player_ids: list[int], before) -> dict[int, float]:
    lv = cfg["live"]
    rows = session.execute(select(m.PlayerGameLog.player_id, m.PlayerGameLog.pf, m.PlayerGameLog.minutes)
                           .where(m.PlayerGameLog.player_id.in_(player_ids), m.PlayerGameLog.game_date < before)).all()
    df = pd.DataFrame(rows, columns=["pid", "pf", "min"]).groupby("pid").sum()
    prior = lv["pf_rate_prior_min"]
    return {int(pid): (r["pf"] + lv["pf_rate_league"] * prior) / (r["min"] + prior) for pid, r in df.iterrows()}


class LiveEngine:
    def __init__(self, session: Session, cfg: dict, is_mock: bool = False, log_flags: bool = True,
                 record: bool = True, source: str = "run"):
        self.session, self.cfg = session, cfg
        self.p = {**cfg["model"]["points"], **cfg["live"]}
        self.rng = np.random.default_rng(cfg["model"]["random_seed"])
        self.is_mock, self.log_flags, self.record, self.source = is_mock, log_flags, record, source
        self.resolver = PlayerResolver(session)
        self.ctx: dict[str, GameContext] = {}
        self.last_run: dict[str, datetime] = {}
        self.points_seen: dict[int, tuple[int, list[datetime]]] = {}   # player -> (last pts, times it changed)
        self.prev_pf: dict[int, int] = {}            # event detection: last fouls seen per player…
        self.prev_margin: dict[str, int] = {}        # …last lead seen per game…
        self.blowout_fired: set[str] = set()         # …and games whose blowout event already fired
        self.log_routine = True                      # False in event replays (routine flags were logged live)
        self.version = model_version(self.p)

    # ------------------------------------------------------------------ pre-game context

    def context(self, game_id: str, day, at: datetime, quotes: list[PropQuote] | None = None) -> GameContext | None:
        if game_id in self.ctx:
            return self.ctx[game_id]
        quotes = quotes if quotes is not None else quotes_at(self.session, game_id, at)
        if not quotes:
            return None
        res = run_pregame(self.session, self.cfg, day, quotes, [], at, log_flags=False, record=False)
        if res.projections.empty:
            return None
        pre = res.projections[res.projections["game_id"] == game_id].set_index("player_id")
        rates = foul_rates(self.session, self.cfg, [int(i) for i in pre.index], day)
        pre = pre.assign(pf_rate=[rates.get(int(i), self.cfg["live"]["pf_rate_league"]) for i in pre.index])
        ctx = GameContext(game_id, game_id.split("_")[2], pre)
        self.ctx[game_id] = ctx
        return ctx

    # ------------------------------------------------------------------ one update

    def on_state(self, state: LiveGameState, game_id: str, now: datetime, feed_stale: bool,
                 quotes: list[PropQuote] | None = None,
                 history: dict[str, list[QuotePoint]] | None = None) -> pd.DataFrame:
        """Price the game's contracts for this state. Returns one row per contract priced.
        `quotes`/`history` (replay) override reading Kalshi quotes from the database."""
        if state.status != "in" or state.period is None:      # no game clock yet: nothing to price
            return pd.DataFrame()
        ctx = self.context(game_id, state.game_date, now, quotes)
        if ctx is None:
            return pd.DataFrame()

        # who is who: ESPN ids -> our ids
        espn_to_pid = {pl.provider_player_id: self.resolver.from_espn(pl.provider_player_id, pl.name, pl.team)
                       for pl in state.players}
        box = {espn_to_pid[pl.provider_player_id]: pl for pl in state.players if not pl.did_not_play}
        elapsed = elapsed_minutes(state.period or 1, state.clock_s)

        # Events re-price at once; otherwise a routine run at most every `every_s` seconds.
        events = self.detect_events(game_id, state, box, ctx, elapsed, now)
        last = self.last_run.get(game_id)
        routine = last is None or (now - last).total_seconds() >= self.cfg["live"]["every_s"]
        if not routine and not events:
            return pd.DataFrame()
        if routine:
            self.last_run[game_id] = now
        team_of = {pl.provider_player_id: pl.team for pl in state.players}
        starters: dict[str, set[str]] = {}
        for pl in state.players:
            if pl.starter:
                starters.setdefault(pl.team, set()).add(pl.provider_player_id)
        floor = replay_states(state.plays, starters, team_of, [elapsed])[-1].players
        floor_by_pid = {espn_to_pid[e]: s for e, s in floor.items() if e in espn_to_pid}

        ids = [pid for pid in ctx.pre.index if pid in box]
        if not ids:
            return pd.DataFrame()
        pre = ctx.pre.loc[ids]
        margin_home = (state.home_score or 0) - (state.away_score or 0)
        pts = np.array([box[i].pts or 0 for i in ids], float)
        x = LiveInputs(
            pts=pts, fouls=np.array([box[i].pf or 0 for i in ids], float),
            minutes=np.array([box[i].minutes or 0.0 for i in ids], float),
            on_court=np.array([floor_by_pid[i].on_court if i in floor_by_pid else True for i in ids]),
            off_for=np.array([0.0 if i not in floor_by_pid or floor_by_pid[i].on_court or floor_by_pid[i].last_out is None
                              else elapsed - floor_by_pid[i].last_out for i in ids]),
            margin=np.array([margin_home if box[i].team == ctx.home else -margin_home for i in ids], float),
            rate_pre=pre["rate"].to_numpy(float), min_pre=pre["min_mu"].to_numpy(float),
            min_sd=pre["min_sd"].to_numpy(float), pf_rate=pre["pf_rate"].to_numpy(float), elapsed=elapsed,
            period=state.period or 1, period_left=(state.clock_s or 0) / 60)
        final, parts = simulate(x, self.p, self.cfg["live"]["n_sims"], self.rng, return_parts=True)
        for k, pid in enumerate(ids):                       # remember when each player's points changed
            prev = self.points_seen.get(pid)
            if prev is None:
                self.points_seen[pid] = (int(pts[k]), [])
            elif int(pts[k]) != prev[0]:
                self.points_seen[pid] = (int(pts[k]), prev[1] + [now])

        quotes = quotes if quotes is not None else quotes_at(self.session, game_id, now)
        pid_of = {q.ticker: self.resolver.from_name(q.player_name, q.team) for q in quotes}
        clock = f"Q{state.period} {int((state.clock_s or 0) // 60):02d}:{int((state.clock_s or 0) % 60):02d}"
        return self._price(ctx, game_id, ids, final, parts, x, quotes, pid_of, now, feed_stale, clock, history,
                           routine=routine, events=events)

    # ------------------------------------------------------------------ events

    def detect_events(self, game_id: str, state: LiveGameState, box: dict, ctx: GameContext, elapsed: float,
                      now: datetime) -> dict[int, dict]:
        """New events in this update -> {player id: trigger} for every player whose props they affect.
          foul4    a player's personal fouls go from ≤3 to ≥4: him ("fouler") and his teammates ("teammate")
          blowout  the lead first reaches `repricing.blowout_lead` in the second half (from below it):
                   starters ("starter", lose minutes) and bench players ("bench", gain them)
        The first update of a game only records the state (a restart mid-game isn't an event)."""
        ev = self.cfg["events"]
        scope: dict[int, dict] = {}
        first_look = game_id not in self.prev_margin
        for pid, line in box.items():
            pf, prev = int(line.pf or 0), self.prev_pf.get(pid)
            self.prev_pf[pid] = pf
            if "foul4" in ev["types"] and prev is not None and prev < 4 <= pf and not first_look:
                info = {"type": "foul4", "event_player": int(pid), "event_time": now.isoformat(),
                        "clock": f"Q{state.period}", "elapsed": elapsed}
                scope[pid] = {**info, "role": "fouler"}
                for mate, ml in box.items():
                    if mate != pid and ml.team == line.team and mate not in scope:
                        scope[mate] = {**info, "role": "teammate"}
        lead = abs((state.home_score or 0) - (state.away_score or 0))
        prev_lead = self.prev_margin.get(game_id)
        self.prev_margin[game_id] = lead
        threshold = self.cfg["repricing"]["blowout_lead"]
        if ("blowout" in ev["types"] and not first_look and elapsed >= 24 and lead >= threshold
                and prev_lead is not None and prev_lead < threshold and game_id not in self.blowout_fired):
            self.blowout_fired.add(game_id)
            info = {"type": "blowout", "event_time": now.isoformat(), "lead": lead, "clock": f"Q{state.period}",
                    "elapsed": elapsed}
            for pid in box:
                if pid in ctx.pre.index:
                    role = "starter" if ctx.pre.loc[pid, "min_mu"] >= self.p["starter_minutes"] - ev["starter_margin_min"] \
                        else "bench"
                    scope.setdefault(pid, {**info, "role": role})
        if scope:
            log.info("event in %s: %s", game_id, sorted({(t["type"], t.get("event_player")) for t in scope.values()},
                                                         key=str))
        return scope

    # ------------------------------------------------------------------ pricing + flags

    def _price(self, ctx, game_id, ids, final, parts, x, quotes, pid_of, now, feed_stale, clock,
               history=None, routine: bool = True, events: dict | None = None) -> pd.DataFrame:
        """Routine run: every contract, candidates logged as kind "live". Event run: contracts of players in
        an event's scope, candidates logged as kind "event" with the trigger, event thresholds, and no
        price-move suppression (the market reacting is exactly what we are racing)."""
        events = events or {}
        cfg, ed, fees = self.cfg, self.cfg["edge"], self.cfg["odds"]["kalshi"]["fees"]
        row_of = {pid: k for k, pid in enumerate(ids)}
        nf = cfg["edge"]["no_flag"]
        window = max(nf["after_price_move_s"], nf["after_game_move_s"]) + nf["price_move_lookback_s"] + 60
        if history is None:
            hist_rows = self.session.execute(select(m.OddsSnapshot.ticker, m.OddsSnapshot.first_seen_at,
                                                    m.OddsSnapshot.yes_bid, m.OddsSnapshot.yes_ask).where(
                m.OddsSnapshot.ticker.in_([q.ticker for q in quotes]),
                m.OddsSnapshot.last_seen_at >= now - timedelta(seconds=window), m.OddsSnapshot.first_seen_at <= now)).all()
            history = {}
            for t, seen, b, a in hist_rows:
                history.setdefault(t, []).append(QuotePoint(utc(seen), b, a))
        hist: dict[int, dict[str, list[QuotePoint]]] = {}
        for t, pts_ in history.items():
            pid = pid_of.get(t)
            if pid is not None:
                hist.setdefault(pid, {})[t] = [q for q in pts_ if q.t <= now]
        game_reason = game_move_reason(hist, now, cfg)
        preds, cands, ev_cands = [], [], []
        ev_cfg = cfg["events"]
        for q in quotes:
            pid = pid_of.get(q.ticker)
            if pid not in row_of or (not routine and pid not in events):
                continue
            k = row_of[pid]
            if x.pts[k] > q.line:                 # already decided
                continue
            p_yes = float((final[k] > q.line).mean())
            conf = ctx.pre.loc[pid, "confidence"]
            preds.append(dict(ticker=q.ticker, player_id=pid, line=q.line, p_yes=p_yes, conf=conf,
                              mean=float(final[k].mean()), yes_bid=q.yes_bid, yes_ask=q.yes_ask))
            for side in price_both_sides(p_yes, q.yes_bid, q.yes_ask, cfg["odds"]["kalshi"]["order_contracts"],
                                         fees["taker_coef"], 1.0, fees["round_to"]):
                if side.edge < ed["log_min_edge"]:
                    continue
                base_inputs = {"clock": clock, "pts": float(x.pts[k]), "fouls": float(x.fouls[k]),
                               "minutes": float(x.minutes[k]), "on_court": bool(x.on_court[k]),
                               "blowout_meter": float(parts["blowout_meter"][k]), "p_gone": float(parts["p_gone"][k]),
                               "rate": float(parts["rate"][k]), "minutes_left_mean": float(parts["minutes_left_mean"][k]),
                               "quote": {"yes_bid": q.yes_bid, "yes_ask": q.yes_ask, "yes_bid_size": q.yes_bid_size,
                                         "yes_ask_size": q.yes_ask_size, "fetched_at": q.fetched_at.isoformat()},
                               "model_version": self.version}
                common_reasons: list[Reason] = []
                if feed_stale:
                    common_reasons.append(Reason("feed_lagging", "our play-by-play is behind"))
                if conf == "low":
                    common_reasons.append(Reason("low_confidence", ctx.pre.loc[pid, "low_reasons"]))
                if (r := scoring_window_reason(self.points_seen.get(pid, (0, []))[1], now, cfg)):
                    common_reasons.append(r)
                common_reasons += market_reasons(q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size, side.side,
                                                 q.fetched_at, now, cfg)
                row = dict(ticker=q.ticker, player_id=pid, game_id=game_id, line=q.line, side=side.side,
                           model_prob=side.model_prob, ask=side.ask, fee=side.fee, cost=side.cost, edge=side.edge,
                           ev_per_dollar=side.ev_per_dollar, confidence=conf)
                if pid in events:
                    ev_cands.append(dict(row, passes=side.edge >= ev_cfg["min_edge"]
                                         and side.ev_per_dollar >= ev_cfg["min_ev_per_dollar"],
                                         reasons=[r.as_dict() for r in common_reasons],
                                         inputs={**base_inputs, "trigger": events[pid]}))
                if not routine:
                    continue
                reasons: list[Reason] = list(common_reasons)
                if (r := player_move_reason(hist.get(pid, {}), now, cfg)):
                    reasons.append(r)
                if game_reason:
                    reasons.append(game_reason)
                passes = side.edge >= ed["min_edge"] and side.ev_per_dollar >= ed["min_ev_per_dollar"]
                cands.append(dict(row, passes=passes, reasons=[r.as_dict() for r in reasons], inputs=base_inputs))
        if self.record and routine:
            for r in preds:
                self.session.add(m.ModelPrediction(created_at=now, kind="live", source=self.source, game_id=game_id,
                                                   player_id=int(r["player_id"]), ticker=r["ticker"], line=r["line"],
                                                   p_yes=r["p_yes"], proj_mean=r["mean"], confidence=r["conf"],
                                                   model_version=self.version, game_clock=clock, is_mock=self.is_mock))
        c = pd.DataFrame(cands)
        if not c.empty:
            c["status"] = assign_statuses(c, ed["one_flag_per_player"])
            if self.log_flags and self.log_routine:
                self._log(c, now)
        e = pd.DataFrame(ev_cands)
        if not e.empty:
            e["status"] = assign_statuses(e, ed["one_flag_per_player"])
            if self.log_flags:
                self._log(e, now, kind="event")
        self.session.commit()
        return pd.DataFrame(preds).assign(clock=clock, at=now)

    def _log(self, c: pd.DataFrame, now: datetime, kind: str = "live") -> None:
        cooldown = timedelta(seconds=self.cfg["edge"]["live_flag_cooldown_s"])
        recent = set() if kind == "event" else {(f.ticker, f.side, f.status) for f in self.session.scalars(
            select(m.Flag).where(m.Flag.kind == kind, m.Flag.created_at >= now - cooldown, m.Flag.created_at <= now))}
        for r in c.to_dict("records"):
            if (r["ticker"], r["side"], r["status"]) in recent:
                continue
            self.session.add(m.Flag(created_at=now, kind=kind, status=r["status"], reasons=r["reasons"], is_paper=True,
                                    game_id=r["game_id"], player_id=int(r["player_id"]), ticker=r["ticker"], stat="pts",
                                    line=r["line"], side=r["side"], model_prob=r["model_prob"], ask=r["ask"],
                                    fee_per_contract=r["fee"], cost=r["cost"], edge=r["edge"],
                                    ev_per_dollar=r["ev_per_dollar"], confidence=r["confidence"], inputs=r["inputs"],
                                    is_mock=self.is_mock))
