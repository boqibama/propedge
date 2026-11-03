"""Event-triggered flags: did Kalshi move our way over the next minutes, and could we have traded?

For every event candidate (blowout onset, 4th foul — the player himself and his teammates):
  move at h seconds = Kalshi mid for our side at t + h  −  mid at t      (did the market come our way?)
  CLV at h seconds  = Kalshi mid for our side at t + h  −  the ask we'd have paid at t
                      (also pays for crossing the spread: what a trade would actually have captured)
  tradability at t  = spread, contracts offered at our ask, dollars that buys;
                      over the next 5 minutes: the tightest spread seen
plus the final result and paper P&L per contract (after fee).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.db import models as m
from propedge.db.store import utc
from propedge.edge.scoreboard import final_points, game_date_of


def event_rows(session: Session, cfg: dict, since: date | None = None, include_replay: bool = True,
               kind: str = "event", dedupe: bool = False, until: date | None = None) -> pd.DataFrame:
    """kind="event": event candidates. kind="live": routine live candidates at ordinary moments — the
    BASELINE (in-game, the NO side drifts up as the clock runs; events must beat that drift)."""
    q = select(m.Flag).where(m.Flag.kind == kind)
    if not include_replay:
        q = q.where(m.Flag.is_mock.is_(False))
    flags = session.scalars(q).all()
    flags = [f for f in flags if (since is None or game_date_of(f.game_id) >= since)
             and (until is None or game_date_of(f.game_id) <= until)]       # by game night, not UTC date
    if dedupe:      # routine candidates repeat every few minutes: keep the first per contract/side
        seen, keep = set(), []
        for f in sorted(flags, key=lambda f: f.created_at):
            if (f.ticker, f.side) not in seen:
                seen.add((f.ticker, f.side))
                keep.append(f)
        flags = keep
    if not flags:
        return pd.DataFrame()
    horizons = cfg["events"]["clv_horizons_s"]
    window = max(horizons)
    liq = cfg["odds"]["liquidity"]
    tickers = list({f.ticker for f in flags})
    od = pd.DataFrame(session.execute(select(m.OddsSnapshot.ticker, m.OddsSnapshot.first_seen_at, m.OddsSnapshot.yes_bid,
                                             m.OddsSnapshot.yes_ask, m.OddsSnapshot.yes_bid_size,
                                             m.OddsSnapshot.yes_ask_size).where(m.OddsSnapshot.ticker.in_(tickers))).all(),
                      columns=["ticker", "t", "bid", "ask", "bid_size", "ask_size"])
    od["t"] = od["t"].map(utc)
    od = od.sort_values("t")
    by_ticker = {t: d.reset_index(drop=True) for t, d in od.groupby("ticker")}
    rows = []
    for f in flags:
        t0 = utc(f.created_at)
        trig = (f.inputs or {}).get("trigger", {})
        qt = (f.inputs or {}).get("quote", {})
        d = by_ticker.get(f.ticker)
        row = {"id": f.id, "t": t0, "game_id": f.game_id, "player_id": f.player_id, "ticker": f.ticker,
               "line": f.line, "side": f.side, "status": f.status, "confidence": f.confidence, "edge": f.edge,
               "ask": f.ask, "cost": f.cost, "type": trig.get("type"), "role": trig.get("role"),
               "replay": bool(f.is_mock), "clock": (f.inputs or {}).get("clock")}
        bid, ask = qt.get("yes_bid"), qt.get("yes_ask")
        row["spread"] = (ask - bid) if bid is not None and ask is not None else np.nan
        mid0 = (bid + ask) / 2 if bid is not None and ask is not None else np.nan
        mid0_side = mid0 if f.side == "yes" else 1 - mid0
        size = qt.get("yes_ask_size") if f.side == "yes" else qt.get("yes_bid_size")
        row["size_at_ask"] = size
        row["dollars_at_ask"] = size * f.ask if size is not None else np.nan
        row["tradable_now"] = bool(row["spread"] == row["spread"] and row["spread"] <= liq["max_spread"] + 1e-9
                                   and size is not None and size >= liq["min_ask_size"])
        if d is not None and len(d):
            ts = pd.DatetimeIndex(pd.to_datetime(d["t"], utc=True))   # sorted
            for h in horizons:
                i = int(ts.searchsorted(pd.Timestamp(t0 + timedelta(seconds=h)), side="right")) - 1   # quote at t0 + h
                if i >= 0 and pd.notna(d.loc[i, "bid"]) and pd.notna(d.loc[i, "ask"]):
                    mid_yes = (d.loc[i, "bid"] + d.loc[i, "ask"]) / 2
                    mid_side = mid_yes if f.side == "yes" else 1 - mid_yes
                    row[f"clv_{h}s"] = mid_side - f.ask              # incl. the cost of crossing the spread
                    row[f"move_{h}s"] = mid_side - mid0_side          # pure price move toward our side
            win = d[(d["t"] >= t0) & (d["t"] <= t0 + timedelta(seconds=window))].dropna(subset=["bid", "ask"])
            row["min_spread_5m"] = float((win["ask"] - win["bid"]).min()) if len(win) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    pts = final_points(session, df["game_id"].unique().tolist())
    df = df.merge(pts, on=["player_id", "game_id"], how="left")
    yes = df["pts"] > df["line"]
    df["won"] = np.where(df["pts"].isna(), np.nan, np.where(df["side"] == "yes", yes, ~yes))
    df["pnl"] = np.where(df["pts"].isna(), np.nan, np.where(df["won"] == 1, 1 - df["cost"], -df["cost"]))
    return df


def event_summary(df: pd.DataFrame, horizons: list[int], by=("type", "role", "side")) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(list(by), dropna=False)
    out = pd.DataFrame({"n": g.size()})
    for h in horizons:
        mv, c = f"move_{h}s", f"clv_{h}s"
        if mv in df:
            out[f"move_{h // 60}m"] = g[mv].mean()
            out[f"moved_our_way_{h // 60}m"] = g[mv].apply(lambda s: float((s.dropna() > 0).mean()) if s.notna().any()
                                                          else np.nan)
        if c in df:
            out[f"clv_{h // 60}m"] = g[c].mean()
    out["win_rate"] = g["won"].mean()
    out["pnl_per_contract"] = g["pnl"].mean()
    out["median_spread"] = g["spread"].median()
    out["median_size_at_ask"] = g["size_at_ask"].median()
    out["tradable_now"] = g["tradable_now"].mean()
    out["min_spread_5m"] = g["min_spread_5m"].median()
    return out.reset_index()
