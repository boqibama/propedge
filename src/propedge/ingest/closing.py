"""Kalshi's 'closing line': the last quote for each contract before tip-off.

Tip-off time = the first play's wall-clock time in our play-by-play (the jump ball), else our first
in-progress snapshot, else the scheduled start. Two sources:
- "poll":    our own odds_snapshots (exact quotes we saw; used for every night we poll)
- "candles": Kalshi's public 1-minute bid/ask candles (backfill for games we didn't poll)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from propedge.db import models as m
from propedge.db.store import upsert, utc
from propedge.http import HttpClient
from propedge.models.suppression import mid

log = logging.getLogger(__name__)


def tip_time(session: Session, game_id: str) -> datetime | None:
    first_play = session.scalar(select(func.min(m.Play.wallclock)).where(m.Play.game_id == game_id))
    if first_play:
        return utc(first_play)
    first_live = session.scalar(select(func.min(m.LiveGameSnapshot.fetched_at)).where(
        m.LiveGameSnapshot.game_id == game_id, m.LiveGameSnapshot.status == "in"))
    if first_live:
        return utc(first_live)
    game = session.get(m.Game, game_id)
    return utc(game.start_time) if game and game.start_time else None


def capture_from_snapshots(session: Session, game_id: str, tip: datetime, max_age_min: float = 20) -> int:
    """Closing quote per contract from our own snapshots: the quote that was current at `tip`, but only
    if we had confirmed it within `max_age_min` of tip-off (otherwise use candles — see stale_or_missing)."""
    tickers = session.scalars(select(m.PropMarket.ticker).where(m.PropMarket.game_id == game_id)).all()
    rows = []
    for t in tickers:
        snap = session.scalars(select(m.OddsSnapshot).where(m.OddsSnapshot.ticker == t,
                                                            m.OddsSnapshot.first_seen_at <= tip)
                               .order_by(m.OddsSnapshot.first_seen_at.desc()).limit(1)).first()
        if snap is None or utc(snap.last_seen_at) < tip - timedelta(minutes=max_age_min):
            continue
        rows.append(dict(ticker=t, game_id=game_id, tip_time=tip, quote_time=min(utc(snap.last_seen_at), tip),
                         yes_bid=snap.yes_bid, yes_ask=snap.yes_ask, mid=mid(snap.yes_bid, snap.yes_ask), source="poll"))
    upsert(session, m.ClosingQuote, rows, ["ticker"])
    session.commit()
    return len(rows)


def capture_from_candles(session: Session, client: HttpClient, base_url: str, series: str, game_id: str,
                         tickers: list[str], tip: datetime, lookback_h: float = 6) -> int:
    """Closing quote from Kalshi's 1-minute candles: the last candle that ended at or before tip."""
    rows = []
    for t in tickers:
        r = client.get_json(f"{base_url}/series/{series}/markets/{t}/candlesticks",
                            params={"start_ts": int((tip - timedelta(hours=lookback_h)).timestamp()),
                                    "end_ts": int(tip.timestamp()), "period_interval": 1})
        if not r.ok:
            log.warning("candles %s: %s", t, r.error)
            continue
        candles = [c for c in r.data.get("candlesticks", []) if c["end_period_ts"] <= tip.timestamp()]
        if not candles:
            continue
        c = candles[-1]
        bid = float(c["yes_bid"]["close_dollars"]) or None
        ask = float(c["yes_ask"]["close_dollars"])
        ask = ask if 0 < ask < 1 else None
        rows.append(dict(ticker=t, game_id=game_id, tip_time=tip,
                         quote_time=datetime.fromtimestamp(c["end_period_ts"], tip.tzinfo),
                         yes_bid=bid, yes_ask=ask, mid=mid(bid, ask), source="candles"))
    upsert(session, m.ClosingQuote, rows, ["ticker"])
    session.commit()
    return len(rows)


def stale_or_missing(session: Session, game_id: str, max_age_min: float = 20) -> list[str]:
    """Contracts of a game whose closing quote is missing or was last confirmed long before tip-off."""
    tickers = session.scalars(select(m.PropMarket.ticker).where(m.PropMarket.game_id == game_id)).all()
    have = {c.ticker: c for c in session.scalars(select(m.ClosingQuote).where(m.ClosingQuote.game_id == game_id))}
    out = []
    for t in tickers:
        c = have.get(t)
        if c is None or c.quote_time is None or utc(c.quote_time) < utc(c.tip_time) - timedelta(minutes=max_age_min):
            out.append(t)
    return out
