"""Model vs market scoreboard, and pre-game closing-line value (CLV).

Scoreboard: for every contract with a closing quote and a settled result, compare
    - our last pre-game probability (made before tip-off),
    - Kalshi's closing mid price,
    - a blend  w·model + (1 − w)·market, with w fitted on the settled data so far,
using Brier score and log loss (lower is better). Beating Kalshi's mid is the real test.

CLV: for every logged candidate, the price we'd have paid when we logged it vs the closing mid for
the same side. Positive CLV = the market moved toward us before tip-off — the fastest signal that a
disagreement was right rather than model error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.db import models as m
from propedge.db.store import utc
from propedge.models.scoring import brier, log_loss

log = logging.getLogger(__name__)

def fit_blend_weight(p_model: np.ndarray, p_market: np.ndarray, y: np.ndarray) -> float:
    """Weight on the model (0 = market only, 1 = model only) that minimises log loss."""
    grid = np.linspace(0, 1, 21)
    losses = [log_loss(w * p_model + (1 - w) * p_market, y) for w in grid]
    return float(grid[int(np.argmin(losses))])


# ----------------------------------------------------------------------------- data

def game_date_of(game_id: str) -> date:
    """US-Eastern game date from our game id ("2026-10-24_WAS_CHI"). Nights are filtered by this, never
    by the UTC timestamp: a 7 pm PT game is already the next day in UTC."""
    return date.fromisoformat(game_id[:10])


def in_nights(game_ids: pd.Series, since: date | None, until: date | None) -> pd.Series:
    d = game_ids.map(game_date_of)
    keep = pd.Series(True, index=game_ids.index)
    if since:
        keep &= d >= since
    if until:
        keep &= d <= until
    return keep


def final_points(session: Session, game_ids: list[str]) -> pd.DataFrame:
    """Final points per (player_id, game_id): official NBA logs, else the final ESPN box score we
    polled (available right after the game; NBA logs can take hours). Players who didn't play are absent.
    If we never stored a "post" snapshot (the poller stopped at 0:00 of Q4), the last box score counts only
    when its game score equals the final score ESPN's scoreboard gave."""
    logs = pd.DataFrame(session.execute(select(m.PlayerGameLog.player_id, m.PlayerGameLog.game_id, m.PlayerGameLog.pts)
                                        .where(m.PlayerGameLog.game_id.in_(game_ids))).all(),
                        columns=["player_id", "game_id", "pts"])
    missing = [g for g in game_ids if g not in set(logs["game_id"])]
    rows = []
    for gid in missing:
        last = session.scalar(select(m.LiveGameSnapshot.id).where(m.LiveGameSnapshot.game_id == gid,
                                                                  m.LiveGameSnapshot.status == "post")
                              .order_by(m.LiveGameSnapshot.id.desc()).limit(1))
        box_snap = session.scalar(select(m.LivePlayerSnapshot.snapshot_id).join(
            m.LiveGameSnapshot, m.LiveGameSnapshot.id == m.LivePlayerSnapshot.snapshot_id)
            .where(m.LiveGameSnapshot.game_id == gid).order_by(m.LivePlayerSnapshot.snapshot_id.desc()).limit(1))
        if box_snap is None:
            continue
        if last is None:
            game = session.get(m.Game, gid)
            final = session.get(m.LiveGameSnapshot, box_snap)
            if (game is None or game.status != "post" or final is None or game.home_score is None
                    or (final.home_score, final.away_score) != (game.home_score, game.away_score)):
                continue
        for b in session.scalars(select(m.LivePlayerSnapshot).where(m.LivePlayerSnapshot.snapshot_id == box_snap)):
            if b.minutes and b.minutes > 0 and not b.did_not_play:
                rows.append({"player_id": b.player_id, "game_id": gid, "pts": b.pts})
    return pd.concat([logs, pd.DataFrame(rows, columns=["player_id", "game_id", "pts"])], ignore_index=True)


def settled_rungs(session: Session, since: date | None = None, until: date | None = None) -> pd.DataFrame:
    """One row per contract with: closing mid, our last pre-game prediction before tip, outcome."""
    q = (select(m.ClosingQuote.ticker, m.ClosingQuote.game_id, m.ClosingQuote.tip_time, m.ClosingQuote.mid,
                m.ClosingQuote.source, m.PropMarket.line, m.PropMarket.player_id, m.Game.game_date)
         .join(m.PropMarket, m.PropMarket.ticker == m.ClosingQuote.ticker)
         .join(m.Game, m.Game.id == m.ClosingQuote.game_id))
    if since:
        q = q.where(m.Game.game_date >= since)
    if until:
        q = q.where(m.Game.game_date <= until)
    close = pd.DataFrame(session.execute(q).all(),
                         columns=["ticker", "game_id", "tip_time", "mid", "close_source", "line", "player_id", "game_date"])
    if close.empty:
        return close
    preds = pd.DataFrame(session.execute(
        select(m.ModelPrediction.ticker, m.ModelPrediction.created_at, m.ModelPrediction.p_yes,
               m.ModelPrediction.confidence, m.ModelPrediction.source, m.ModelPrediction.model_version)
        .where(m.ModelPrediction.kind == "pregame", m.ModelPrediction.ticker.in_(close["ticker"].tolist()))).all(),
        columns=["ticker", "created_at", "p_model", "confidence", "pred_source", "model_version"])
    if preds.empty:
        return pd.DataFrame()
    preds["created_at"] = preds["created_at"].map(utc)
    close["tip_time"] = close["tip_time"].map(utc)
    merged = preds.merge(close[["ticker", "tip_time"]], on="ticker")
    merged = merged[merged["created_at"] <= merged["tip_time"]]
    last = merged.sort_values("created_at").groupby("ticker").tail(1)
    out = close.merge(last.drop(columns=["tip_time"]), on="ticker")
    pts = final_points(session, out["game_id"].unique().tolist())
    out = out.merge(pts, on=["player_id", "game_id"], how="left")
    out = out[out["pts"].notna() & out["mid"].notna()]       # did-not-play settles at a fair price: excluded
    out["y"] = (out["pts"] > out["line"]).astype(float)
    out["rung"] = (out["line"] + 0.5).astype(int).astype(str) + "+"
    return out.reset_index(drop=True)


@dataclass
class Board:
    rows: pd.DataFrame
    blend_weight: float
    table: pd.DataFrame          # one row per group


def scoreboard(df: pd.DataFrame) -> Board:
    w = fit_blend_weight(df["p_model"].to_numpy(), df["mid"].to_numpy(), df["y"].to_numpy())
    df = df.assign(p_blend=w * df["p_model"] + (1 - w) * df["mid"])

    def row(name, d):
        y = d["y"].to_numpy()
        return {"group": name, "n": len(d), "hit_rate": float(y.mean()) if len(d) else float("nan"),
                **{f"brier_{k}": brier(d[c].to_numpy(), y) for k, c in
                   (("model", "p_model"), ("kalshi", "mid"), ("blend", "p_blend"))},
                **{f"logloss_{k}": log_loss(d[c].to_numpy(), y) for k, c in
                   (("model", "p_model"), ("kalshi", "mid"), ("blend", "p_blend"))}}

    rows = [row("ALL", df)]
    for conf, d in df.groupby("confidence"):
        rows.append(row(f"confidence={conf}", d))
    for rung, d in sorted(df.groupby("rung"), key=lambda kv: int(kv[0][:-1])):
        rows.append(row(f"line {rung}", d))
    return Board(df, w, pd.DataFrame(rows))


# ----------------------------------------------------------------------------- CLV

def flag_clv(session: Session, since: date | None = None, until: date | None = None) -> pd.DataFrame:
    """First logged occurrence of each (contract, side) candidate, with closing-line value:
        clv = closing mid for our side − the ask we'd have paid   (in dollars per contract)."""
    q = (select(m.Flag.id, m.Flag.created_at, m.Flag.kind, m.Flag.status, m.Flag.ticker, m.Flag.side, m.Flag.ask,
                m.Flag.cost, m.Flag.edge, m.Flag.confidence, m.Flag.game_id, m.ClosingQuote.mid, m.ClosingQuote.tip_time,
                m.Game.game_date)
         .join(m.ClosingQuote, m.ClosingQuote.ticker == m.Flag.ticker)
         .join(m.Game, m.Game.id == m.Flag.game_id)
         .where(m.Flag.kind == "pregame"))
    if since:
        q = q.where(m.Game.game_date >= since)
    if until:
        q = q.where(m.Game.game_date <= until)
    df = pd.DataFrame(session.execute(q).all(), columns=["id", "created_at", "kind", "status", "ticker", "side", "ask",
                                                         "cost", "edge", "confidence", "game_id", "close_mid",
                                                         "tip_time", "game_date"])
    if df.empty:
        return df
    df["created_at"] = df["created_at"].map(utc)
    df["tip_time"] = df["tip_time"].map(utc)
    df = df[df["created_at"] <= df["tip_time"]].dropna(subset=["close_mid"])
    df = df.sort_values("created_at").groupby(["ticker", "side"]).head(1)
    close_side = np.where(df["side"] == "yes", df["close_mid"], 1 - df["close_mid"])
    df["clv"] = close_side - df["ask"]
    df["clv_after_fee"] = close_side - df["cost"]
    df["minutes_before_tip"] = (df["tip_time"] - df["created_at"]).dt.total_seconds() / 60
    return df.reset_index(drop=True)


def clv_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["status", "side"])
    return pd.DataFrame({"n": g.size(), "avg_clv": g["clv"].mean(), "avg_clv_after_fee": g["clv_after_fee"].mean(),
                         "share_beat_close": g["clv"].apply(lambda s: float((s > 0).mean())),
                         "avg_edge_claimed": g["edge"].mean()}).reset_index()


def side_split(session: Session, since: date | None = None, until: date | None = None) -> pd.DataFrame:
    """Flags by side per night — a steady lean to one side means bias, not edge."""
    q = select(m.Game.game_date, m.Flag.side, m.Flag.status, m.Flag.ticker).join(m.Game, m.Game.id == m.Flag.game_id) \
        .where(m.Flag.kind == "pregame", m.Flag.status == "flag")
    if since:
        q = q.where(m.Game.game_date >= since)
    if until:
        q = q.where(m.Game.game_date <= until)
    df = pd.DataFrame(session.execute(q).all(), columns=["game_date", "side", "status", "ticker"])
    if df.empty:
        return df
    df = df.drop_duplicates(["ticker", "side"])
    return df.pivot_table(index="game_date", columns="side", values="ticker", aggfunc="count", fill_value=0)


# ----------------------------------------------------------------------------- backfill for past nights

def retro_day(session: Session, cfg: dict, day: date, kalshi_client, base_url: str) -> dict:
    """Rebuild a past night: Kalshi's contracts for that day, closing quotes from Kalshi's 1-minute
    candles, and our pre-game predictions as of that morning (source="retro", stamped 1 minute
    before the earliest tip so they count as pre-game)."""
    from propedge.edge.pregame import actual_absences, run_pregame
    from propedge.ingest.closing import capture_from_candles, tip_time
    from propedge.ingest.live import LiveStore
    from propedge.odds.kalshi import parse_market

    series = cfg["odds"]["kalshi"]["series"]["pts"]
    stamp = day.strftime("%y%b%d").upper()
    quotes, cursor = [], None
    for status in ("settled", "closed", "open"):
        cursor = None
        while True:
            params = {"series_ticker": series, "status": status, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            r = kalshi_client.get_json(f"{base_url}/markets", params=params)
            if not r.ok:
                break
            for mk in r.data.get("markets", []):
                if f"-{stamp}" in mk["event_ticker"]:
                    q = parse_market(mk, "pts", r.fetched_at)
                    if q:
                        quotes.append(q)
            cursor = r.data.get("cursor")
            if not cursor:
                break
    if not quotes:
        return {"day": str(day), "contracts": 0}
    LiveStore(session, stale_after_s=cfg["live_feed"]["stale_after_s"]).store_quotes(quotes)
    games = session.scalars(select(m.Game).where(m.Game.game_date == day)).all()
    tips, n_close = {}, 0
    for g in games:
        tip = tip_time(session, g.id)
        if tip is None:
            continue
        tips[g.id] = tip
        tickers = [q.ticker for q in quotes if f"{q.away}{q.home}" == f"{g.away}{g.home}"]
        # Always (re)capture: the tip time may have improved (scheduled -> actual first play) and the
        # closing quote must use the same tip time as the predictions.
        n_close += capture_from_candles(session, kalshi_client, base_url, series, g.id, tickers, tip)
    if not tips:
        return {"day": str(day), "contracts": len(quotes), "closing": 0}
    as_of = min(tips.values()) - timedelta(minutes=1)
    # Re-running a day replaces its retro predictions (a newer model version supersedes them).
    from sqlalchemy import delete
    ids = list(tips)
    session.execute(delete(m.ModelPrediction).where(m.ModelPrediction.source == "retro",
                                                    m.ModelPrediction.game_id.in_(ids)))
    session.execute(delete(m.PlayerProjection).where(m.PlayerProjection.source == "retro",
                                                     m.PlayerProjection.game_id.in_(ids)))
    session.commit()
    res = run_pregame(session, cfg, day, quotes, [], as_of, known_absent_override=actual_absences(session, cfg, day),
                      log_flags=False, record=True, source="retro")
    return {"day": str(day), "contracts": len(quotes), "closing": n_close, "predictions": len(res.rungs)}


# ----------------------------------------------------------------------------- live (real-time)

def live_rows(session: Session, since: date | None = None, until: date | None = None) -> pd.DataFrame:
    """Live predictions made during games (source "run"), each matched to the Kalshi quote that was
    current at that moment, with the final outcome."""
    q = select(m.ModelPrediction.created_at, m.ModelPrediction.game_id, m.ModelPrediction.player_id,
               m.ModelPrediction.ticker, m.ModelPrediction.line, m.ModelPrediction.p_yes,
               m.ModelPrediction.confidence, m.ModelPrediction.game_clock).where(
        m.ModelPrediction.kind == "live", m.ModelPrediction.source == "run")
    pr = pd.DataFrame(session.execute(q).all(), columns=["t", "game_id", "player_id", "ticker", "line", "p_model",
                                                         "confidence", "clock"])
    if pr.empty:
        return pr
    pr["t"] = pd.to_datetime(pr["t"].map(utc))
    pr = pr[in_nights(pr["game_id"], since, until)]
    if pr.empty:
        return pr
    od = pd.DataFrame(session.execute(select(m.OddsSnapshot.ticker, m.OddsSnapshot.first_seen_at, m.OddsSnapshot.yes_bid,
                                             m.OddsSnapshot.yes_ask).where(
        m.OddsSnapshot.ticker.in_(pr["ticker"].unique().tolist()))).all(), columns=["ticker", "t", "bid", "ask"])
    od["t"] = pd.to_datetime(od["t"].map(utc))
    od = od.dropna(subset=["bid", "ask"]).sort_values("t")
    pr = pd.merge_asof(pr.sort_values("t"), od, on="t", by="ticker", direction="backward")
    pr["p_kalshi"] = (pr["bid"] + pr["ask"]) / 2
    pts = final_points(session, pr["game_id"].unique().tolist())
    pr = pr.merge(pts, on=["player_id", "game_id"], how="left").dropna(subset=["pts", "p_kalshi"])
    pr["y"] = (pr["pts"] > pr["line"]).astype(float)
    pr["quarter"] = pd.to_numeric(pr["clock"].str.extract(r"Q(\d+)")[0], errors="coerce")
    pr = pr.dropna(subset=["quarter"])
    pr["quarter"] = pr["quarter"].astype(int).clip(upper=4)
    return pr.reset_index(drop=True)


def live_flag_results(session: Session, since: date | None = None, until: date | None = None) -> pd.DataFrame:
    """Settle live paper flags/candidates: win/loss and P&L per contract (cost incl. fee)."""
    q = select(m.Flag.created_at, m.Flag.status, m.Flag.side, m.Flag.ticker, m.Flag.line, m.Flag.cost, m.Flag.edge,
               m.Flag.player_id, m.Flag.game_id).where(m.Flag.kind == "live")
    df = pd.DataFrame(session.execute(q).all(), columns=["t", "status", "side", "ticker", "line", "cost", "edge",
                                                         "player_id", "game_id"])
    df = df[in_nights(df["game_id"], since, until)]
    if df.empty:
        return df
    df = df.sort_values("t").drop_duplicates(["ticker", "side", "status"])
    pts = final_points(session, df["game_id"].unique().tolist())
    df = df.merge(pts, on=["player_id", "game_id"], how="left").dropna(subset=["pts"])
    yes = df["pts"] > df["line"]
    df["won"] = np.where(df["side"] == "yes", yes, ~yes)
    df["pnl"] = np.where(df["won"], 1 - df["cost"], -df["cost"])
    return df


# ----------------------------------------------------------------------------- model versions vs market

def rebuild_day(session: Session, cfg: dict, day: date) -> pd.DataFrame:
    """Current-model pre-game P(yes) for every contract of a past day, as of that morning (nothing stored)."""
    from propedge.edge.pregame import actual_absences, run_pregame
    from propedge.odds.base import PropQuote

    mk = session.scalars(select(m.PropMarket).join(m.Game, m.Game.id == m.PropMarket.game_id)
                         .where(m.Game.game_date == day)).all()
    quotes = []
    for x in mk:
        d, away, home = x.game_id.split("_")
        quotes.append(PropQuote(provider=x.provider, ticker=x.ticker, event_ticker=x.event_ticker, game_date=day,
                                away=away, home=home, player_name=x.player_name, team=x.team, stat=x.stat,
                                line=x.line, yes_bid=None, yes_ask=None, yes_bid_size=None, yes_ask_size=None,
                                last_price=None, volume=None, open_interest=None, status=x.status,
                                close_time=x.close_time, fetched_at=datetime.now(timezone.utc)))
    if not quotes:
        return pd.DataFrame()
    res = run_pregame(session, cfg, day, quotes, [], datetime.now(timezone.utc),
                      known_absent_override=actual_absences(session, cfg, day), log_flags=False, record=False)
    return res.rungs[["ticker", "model_prob", "confidence"]].rename(columns={"model_prob": "p_new",
                                                                              "confidence": "conf_new"})


def lean_table(df: pd.DataFrame, cols: dict[str, str], gap: float = 0.12) -> pd.DataFrame:
    """For each forecast: Brier, mean (p − Kalshi mid) — positive = leans to overs — and how many contracts
    it disagrees with the market by ≥ `gap` on each side."""
    rows = []
    for name, c in cols.items():
        diff = df[c] - df["mid"]
        rows.append({"forecast": name, "n": len(df), "brier": brier(df[c].to_numpy(), df["y"].to_numpy()),
                     "log_loss": log_loss(df[c].to_numpy(), df["y"].to_numpy()),
                     "mean_p_minus_mid": float(diff.mean()),
                     f"over_disagreements(≥{gap:.2f})": int((diff >= gap).sum()),
                     f"under_disagreements(≥{gap:.2f})": int((diff <= -gap).sum())})
    return pd.DataFrame(rows)
