"""The OddsProvider interface. Every odds source (Kalshi, the replay/mock providers) returns
the same `PropQuote` objects, so the rest of the tool never needs to know where a price came from.

Quotes are exchange-shaped: a YES contract that pays $1 if the player's stat ends ABOVE `line`.
- Buying YES costs `yes_ask`.
- Buying NO costs 1 − `yes_bid` (on Kalshi, a NO bid at x is the same order as a YES ask at 1 − x).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class PropQuote:
    provider: str                # "kalshi", "replay", "mock"
    ticker: str                  # the contract's id at the provider
    event_ticker: str            # groups all contracts of one game
    game_date: date | None       # US-Eastern date, parsed from the event
    away: str | None             # NBA tricodes
    home: str | None
    player_name: str             # as the provider spells it
    team: str | None
    stat: str                    # "pts", ...
    line: float                  # YES wins if stat > line ("25+ points" => 24.5)
    yes_bid: float | None        # dollars, None if nobody is bidding
    yes_ask: float | None        # dollars, None if nobody is offering
    yes_bid_size: float | None   # contracts at the best bid
    yes_ask_size: float | None   # contracts at the best ask
    last_price: float | None
    volume: float | None
    open_interest: float | None
    status: str | None
    close_time: datetime | None
    fetched_at: datetime         # when WE received it (UTC) — the only trustworthy freshness time
    is_mock: bool = False        # True for replay/mock data — shown clearly everywhere

    @property
    def no_ask(self) -> float | None:
        return None if self.yes_bid is None else round(1 - self.yes_bid, 4)

    @property
    def spread(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return round(self.yes_ask - self.yes_bid, 4)


class OddsProvider(ABC):
    name: str
    is_mock: bool = False

    @abstractmethod
    def fetch_quotes(self, stats: list[str]) -> list[PropQuote]:
        """Current quotes for every open market of the given stats (e.g. ["pts"]) across all games."""
