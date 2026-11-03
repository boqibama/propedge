"""Kalshi player-prop markets via the public market-data API (no key needed to read).

Kalshi lists props as ladders: "Stephen Curry: 30+ points" is a YES/NO contract with
floor_strike 29.5 and strike_type "greater" (YES if points > 29.5).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from propedge.config import load_config
from propedge.http import HttpClient, make_client
from propedge.ids import parse_kalshi_event
from propedge.odds.base import OddsProvider, PropQuote

log = logging.getLogger(__name__)


def _price(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_market(m: dict, stat: str, fetched_at: datetime, provider: str = "kalshi") -> PropQuote | None:
    """One Kalshi market dict -> PropQuote. Returns None for markets we can't interpret."""
    if m.get("strike_type") != "greater" or m.get("floor_strike") is None:
        log.debug("skipping %s: strike_type=%s", m.get("ticker"), m.get("strike_type"))
        return None
    title = m.get("title") or m.get("yes_sub_title") or ""
    player_name = title.split(":", 1)[0].strip()
    parsed = parse_kalshi_event(m["event_ticker"])
    game_date, away, home = parsed if parsed else (None, None, None)
    # Player segment of the ticker starts with the team tricode: "...-GSWSCURRY30-35".
    segment = m["ticker"].split("-")[-2] if m["ticker"].count("-") >= 3 else ""
    team = segment[:3] if segment[:3] in {away, home} else None

    bid, ask = _price(m.get("yes_bid_dollars")), _price(m.get("yes_ask_dollars"))
    close = m.get("close_time")
    return PropQuote(
        provider=provider,
        ticker=m["ticker"],
        event_ticker=m["event_ticker"],
        game_date=game_date, away=away, home=home,
        player_name=player_name,
        team=team,
        stat=stat,
        line=float(m["floor_strike"]),
        yes_bid=bid if bid and bid > 0 else None,          # 0 means "no bids"
        yes_ask=ask if ask and ask < 1 else None,          # 1 means "no offers"
        yes_bid_size=_price(m.get("yes_bid_size_fp")),
        yes_ask_size=_price(m.get("yes_ask_size_fp")),
        last_price=_price(m.get("last_price_dollars")),
        volume=_price(m.get("volume_fp")),
        open_interest=_price(m.get("open_interest_fp")),
        status=m.get("status"),
        close_time=datetime.fromisoformat(close.replace("Z", "+00:00")) if close else None,
        fetched_at=fetched_at,
    )


class KalshiOddsProvider(OddsProvider):
    name = "kalshi"

    def __init__(self, client: HttpClient | None = None, config: dict | None = None,
                 on_raw: Callable[[str, str, datetime, dict], None] | None = None):
        cfg = (config or load_config())["odds"]["kalshi"]
        self.on_raw = on_raw          # e.g. a RawRecorder, to save payloads for replay
        self.client = client or make_client("kalshi")
        self.base_url = cfg["base_url"]
        self.series: dict[str, str] = cfg["series"]      # stat -> series ticker
        self.page_size = cfg["page_size"]

    def fetch_quotes(self, stats: list[str]) -> list[PropQuote]:
        quotes: list[PropQuote] = []
        for stat in stats:
            series = self.series.get(stat)
            if not series:
                log.warning("no Kalshi series configured for stat %r", stat)
                continue
            cursor = None
            while True:
                params = {"series_ticker": series, "status": "open", "limit": self.page_size}
                if cursor:
                    params["cursor"] = cursor
                r = self.client.get_json(f"{self.base_url}/markets", params=params)
                if not r.ok:
                    log.warning("Kalshi %s markets failed: %s", series, r.error)
                    break
                if self.on_raw and not cursor:   # replay reads one file per poll: record page 1
                    self.on_raw("kalshi", series, r.fetched_at, r.data)
                for m in r.data.get("markets", []):
                    q = parse_market(m, stat, r.fetched_at)
                    if q:
                        quotes.append(q)
                cursor = r.data.get("cursor")
                if not cursor:
                    break
        return quotes

    def series_fee_info(self, stat: str) -> dict | None:
        """{'fee_type': 'quadratic', 'fee_multiplier': 1} for the stat's series (Phase 4 uses this)."""
        r = self.client.get_json(f"{self.base_url}/series/{self.series[stat]}")
        if not r.ok:
            return None
        s = r.data["series"]
        return {"fee_type": s.get("fee_type"), "fee_multiplier": s.get("fee_multiplier")}

    def orderbook(self, ticker: str) -> dict | None:
        """Full depth for one market: {'yes': [(price, size), ...] best first, 'no': [...]}.
        Kalshi lists bids only; the YES asks are 1 − the NO bids."""
        r = self.client.get_json(f"{self.base_url}/markets/{ticker}/orderbook")
        if not r.ok:
            return None
        book = r.data.get("orderbook_fp") or {}
        def side(levels):
            return sorted(((float(p), float(s)) for p, s in levels or []), reverse=True)
        return {"yes_bids": side(book.get("yes_dollars")), "no_bids": side(book.get("no_dollars")),
                "fetched_at": r.fetched_at}


def load_quotes_file(path, stat: str, series: str, fetched_at: datetime | None = None) -> list[PropQuote]:
    """Quotes from a saved Kalshi markets payload: a recording file ({"fetched_at", "data"}, .json
    or .json.gz) or a raw API page ({"markets": [...]}, then `fetched_at` must be given)."""
    import gzip
    import json
    from pathlib import Path

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        blob = json.load(fh)
    if "data" in blob:
        fetched_at = fetched_at or datetime.fromisoformat(blob["fetched_at"])
        blob = blob["data"]
    if fetched_at is None:
        raise ValueError(f"{path} has no fetched_at; pass the time the quotes were taken")
    quotes = [parse_market(mk, stat, fetched_at) for mk in blob.get("markets", [])]
    return [q for q in quotes if q is not None and q.event_ticker.startswith(series)]
