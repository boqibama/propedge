"""Database schema (SQLAlchemy 2.x). Works on SQLite now and Postgres later.

Every row that describes "the world at a moment" carries the UTC time WE saw it
(`fetched_at` / `first_seen_at`), so staleness and lag can always be measured later.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String,
                        Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ----------------------------------------------------------------------------- reference data

class Team(Base):
    __tablename__ = "teams"
    tricode: Mapped[str] = mapped_column(String(3), primary_key=True)
    nba_team_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    espn_team_id: Mapped[str | None] = mapped_column(String(8), unique=True)
    name: Mapped[str | None] = mapped_column(String(64))


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_player_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    espn_athlete_id: Mapped[str | None] = mapped_column(String(16), unique=True)
    name: Mapped[str] = mapped_column(String(64))
    name_key: Mapped[str] = mapped_column(String(64), index=True)
    team: Mapped[str | None] = mapped_column(String(3))       # most recent team seen
    position: Mapped[str | None] = mapped_column(String(8))


class Game(Base):
    __tablename__ = "games"
    id: Mapped[str] = mapped_column(String(24), primary_key=True)   # "2026-10-21_GSW_LAL"
    nba_game_id: Mapped[str | None] = mapped_column(String(10), unique=True)
    espn_event_id: Mapped[str | None] = mapped_column(String(12), unique=True)
    season: Mapped[str] = mapped_column(String(7), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)       # US-Eastern date
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    away: Mapped[str] = mapped_column(String(3))
    home: Mapped[str] = mapped_column(String(3))
    status: Mapped[str | None] = mapped_column(String(16))          # pre | in | post
    away_score: Mapped[int | None] = mapped_column(Integer)
    home_score: Mapped[int | None] = mapped_column(Integer)
    home_spread: Mapped[float | None] = mapped_column(Float)        # e.g. -5.5 = home favoured by 5.5
    total: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ----------------------------------------------------------------------------- historical stats

class PlayerGameLog(Base):
    __tablename__ = "player_game_logs"
    __table_args__ = (UniqueConstraint("nba_game_id", "nba_player_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_game_id: Mapped[str] = mapped_column(String(10), index=True)
    nba_player_id: Mapped[int] = mapped_column(Integer, index=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), index=True)
    game_id: Mapped[str | None] = mapped_column(ForeignKey("games.id"))
    season: Mapped[str] = mapped_column(String(7), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    team: Mapped[str] = mapped_column(String(3))
    opponent: Mapped[str] = mapped_column(String(3))
    is_home: Mapped[bool] = mapped_column(Boolean)
    minutes: Mapped[float | None] = mapped_column(Float)
    pts: Mapped[int | None] = mapped_column(Integer)
    fgm: Mapped[int | None] = mapped_column(Integer)
    fga: Mapped[int | None] = mapped_column(Integer)
    fg3m: Mapped[int | None] = mapped_column(Integer)
    fg3a: Mapped[int | None] = mapped_column(Integer)
    ftm: Mapped[int | None] = mapped_column(Integer)
    fta: Mapped[int | None] = mapped_column(Integer)
    oreb: Mapped[int | None] = mapped_column(Integer)
    dreb: Mapped[int | None] = mapped_column(Integer)
    reb: Mapped[int | None] = mapped_column(Integer)
    ast: Mapped[int | None] = mapped_column(Integer)
    stl: Mapped[int | None] = mapped_column(Integer)
    blk: Mapped[int | None] = mapped_column(Integer)
    tov: Mapped[int | None] = mapped_column(Integer)
    pf: Mapped[int | None] = mapped_column(Integer)
    plus_minus: Mapped[float | None] = mapped_column(Float)


class TeamGameLog(Base):
    """One row per team per game; enough to estimate possessions (pace) and points allowed."""
    __tablename__ = "team_game_logs"
    __table_args__ = (UniqueConstraint("nba_game_id", "team"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_game_id: Mapped[str] = mapped_column(String(10), index=True)
    game_id: Mapped[str | None] = mapped_column(ForeignKey("games.id"))
    season: Mapped[str] = mapped_column(String(7), index=True)
    game_date: Mapped[date] = mapped_column(Date)
    team: Mapped[str] = mapped_column(String(3), index=True)
    opponent: Mapped[str] = mapped_column(String(3))
    is_home: Mapped[bool] = mapped_column(Boolean)
    minutes: Mapped[float | None] = mapped_column(Float)   # 240 = regulation, more with OT
    pts: Mapped[int | None] = mapped_column(Integer)
    fga: Mapped[int | None] = mapped_column(Integer)
    fta: Mapped[int | None] = mapped_column(Integer)
    oreb: Mapped[int | None] = mapped_column(Integer)
    tov: Mapped[int | None] = mapped_column(Integer)
    pf: Mapped[int | None] = mapped_column(Integer)


class TeamSeasonStats(Base):
    """Season-to-date advanced stats from stats.nba.com, snapshotted on `as_of`."""
    __tablename__ = "team_season_stats"
    __table_args__ = (UniqueConstraint("season", "team", "as_of"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[str] = mapped_column(String(7))
    team: Mapped[str] = mapped_column(String(3))
    as_of: Mapped[date] = mapped_column(Date)
    games: Mapped[int | None] = mapped_column(Integer)
    pace: Mapped[float | None] = mapped_column(Float)
    off_rating: Mapped[float | None] = mapped_column(Float)
    def_rating: Mapped[float | None] = mapped_column(Float)


class Matchup(Base):
    """Who guarded whom (BoxScoreMatchupsV3): one row per offensive player × defender per game."""
    __tablename__ = "matchups"
    __table_args__ = (UniqueConstraint("nba_game_id", "off_nba_player_id", "def_nba_player_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_game_id: Mapped[str] = mapped_column(String(10), index=True)
    off_nba_player_id: Mapped[int] = mapped_column(Integer, index=True)
    def_nba_player_id: Mapped[int] = mapped_column(Integer, index=True)
    matchup_seconds: Mapped[float | None] = mapped_column(Float)
    partial_possessions: Mapped[float | None] = mapped_column(Float)
    off_points: Mapped[int | None] = mapped_column(Integer)
    off_fga: Mapped[int | None] = mapped_column(Integer)
    off_fgm: Mapped[int | None] = mapped_column(Integer)
    off_fg3a: Mapped[int | None] = mapped_column(Integer)


# ----------------------------------------------------------------------------- live data

class LiveGameSnapshot(Base):
    """Game-level state each time the live feed changed (or every heartbeat)."""
    __tablename__ = "live_game_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.id"), index=True)
    provider: Mapped[str] = mapped_column(String(16))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(16))
    period: Mapped[int | None] = mapped_column(Integer)
    clock_s: Mapped[float | None] = mapped_column(Float)         # seconds left in the period
    away_score: Mapped[int | None] = mapped_column(Integer)
    home_score: Mapped[int | None] = mapped_column(Integer)
    n_plays: Mapped[int] = mapped_column(Integer)
    newest_play_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    feed_lag_s: Mapped[float | None] = mapped_column(Float)      # fetched_at − newest play's wall clock
    play_delay_s: Mapped[float | None] = mapped_column(Float)    # median delivery delay of plays new in this fetch
    clock_running: Mapped[bool | None] = mapped_column(Boolean)  # game clock moved since our previous snapshot
    is_stale: Mapped[bool] = mapped_column(Boolean)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)


class LivePlayerSnapshot(Base):
    __tablename__ = "live_player_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("live_game_snapshots.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    team: Mapped[str] = mapped_column(String(3))
    starter: Mapped[bool | None] = mapped_column(Boolean)
    did_not_play: Mapped[bool | None] = mapped_column(Boolean)
    minutes: Mapped[float | None] = mapped_column(Float)
    pts: Mapped[int | None] = mapped_column(Integer)
    fgm: Mapped[int | None] = mapped_column(Integer)
    fga: Mapped[int | None] = mapped_column(Integer)
    fg3m: Mapped[int | None] = mapped_column(Integer)
    fg3a: Mapped[int | None] = mapped_column(Integer)
    ftm: Mapped[int | None] = mapped_column(Integer)
    fta: Mapped[int | None] = mapped_column(Integer)
    reb: Mapped[int | None] = mapped_column(Integer)
    ast: Mapped[int | None] = mapped_column(Integer)
    stl: Mapped[int | None] = mapped_column(Integer)
    blk: Mapped[int | None] = mapped_column(Integer)
    tov: Mapped[int | None] = mapped_column(Integer)
    pf: Mapped[int | None] = mapped_column(Integer)
    plus_minus: Mapped[float | None] = mapped_column(Float)


class Play(Base):
    """Play-by-play. `first_seen_at − wallclock` = how late our feed delivered this play."""
    __tablename__ = "plays"
    __table_args__ = (UniqueConstraint("provider", "provider_play_id"),
                      Index("ix_plays_game_seq", "game_id", "sequence"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(16))
    provider_play_id: Mapped[str] = mapped_column(String(24))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.id"))
    sequence: Mapped[int] = mapped_column(Integer)
    period: Mapped[int] = mapped_column(Integer)
    clock_s: Mapped[float | None] = mapped_column(Float)
    wallclock: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    play_type: Mapped[str] = mapped_column(String(48))
    text: Mapped[str] = mapped_column(Text)
    team: Mapped[str | None] = mapped_column(String(3))
    scoring: Mapped[bool] = mapped_column(Boolean)
    score_value: Mapped[int] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    home_score: Mapped[int | None] = mapped_column(Integer)
    player_ids: Mapped[list] = mapped_column(JSON)   # our players.id, in feed order (shooter/assister, sub in/out)
    # True if the play was already in the feed the first time this poller looked at the game,
    # so first_seen_at says nothing about feed delay for it.
    backlog: Mapped[bool] = mapped_column(Boolean, default=False)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)


# ----------------------------------------------------------------------------- odds

class PropMarket(Base):
    """One tradable contract, e.g. Kalshi 'Stephen Curry: 30+ points'. Static facts only."""
    __tablename__ = "prop_markets"
    ticker: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(16))
    event_ticker: Mapped[str] = mapped_column(String(48), index=True)
    game_id: Mapped[str | None] = mapped_column(String(24), index=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), index=True)
    player_name: Mapped[str] = mapped_column(String(64))
    team: Mapped[str | None] = mapped_column(String(3))
    stat: Mapped[str] = mapped_column(String(8))        # "pts", ...
    line: Mapped[float] = mapped_column(Float)          # YES wins if stat > line (25+ => 24.5)
    close_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str | None] = mapped_column(String(16))
    result: Mapped[str | None] = mapped_column(String(8))
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)


class OddsSnapshot(Base):
    """A quote that was unchanged from first_seen_at to last_seen_at (we re-confirm it on every poll).
    A new row is written only when something in the quote changes."""
    __tablename__ = "odds_snapshots"
    __table_args__ = (Index("ix_odds_ticker_seen", "ticker", "first_seen_at"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("prop_markets.ticker"))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    yes_bid: Mapped[float | None] = mapped_column(Float)
    yes_ask: Mapped[float | None] = mapped_column(Float)
    yes_bid_size: Mapped[float | None] = mapped_column(Float)
    yes_ask_size: Mapped[float | None] = mapped_column(Float)
    last_price: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    open_interest: Mapped[float | None] = mapped_column(Float)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)


# ----------------------------------------------------------------------------- model output & market close

class ModelPrediction(Base):
    """Our probability for every priced contract, every time the model runs (pre-game or live).
    The scoreboard compares the last pre-game prediction with Kalshi's closing price."""
    __tablename__ = "model_predictions"
    __table_args__ = (Index("ix_pred_ticker_time", "ticker", "created_at"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    kind: Mapped[str] = mapped_column(String(8))              # "pregame" | "live"
    source: Mapped[str] = mapped_column(String(12))           # "run" (made at the time) | "retro" (rebuilt as-of)
    game_id: Mapped[str] = mapped_column(String(24), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    ticker: Mapped[str] = mapped_column(String(64))
    line: Mapped[float] = mapped_column(Float)
    p_yes: Mapped[float] = mapped_column(Float)
    proj_mean: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(8))
    model_version: Mapped[str] = mapped_column(String(16))
    game_clock: Mapped[str | None] = mapped_column(String(12))   # live: "Q3 05:12"
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)


class PlayerProjection(Base):
    """Pre-game projection per player-game (used by the rolling bias correction and reports)."""
    __tablename__ = "player_projections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(12))
    game_id: Mapped[str] = mapped_column(String(24), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    proj_minutes: Mapped[float] = mapped_column(Float)
    proj_points: Mapped[float] = mapped_column(Float)
    points_sd: Mapped[float] = mapped_column(Float)
    bias_factor: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[str] = mapped_column(String(8))
    low_reasons: Mapped[str] = mapped_column(String(80))
    model_version: Mapped[str] = mapped_column(String(16))


class ClosingQuote(Base):
    """Kalshi's last quote before tip-off for each contract — the 'closing line'."""
    __tablename__ = "closing_quotes"
    ticker: Mapped[str] = mapped_column(String(64), primary_key=True)
    game_id: Mapped[str] = mapped_column(String(24), index=True)
    tip_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    quote_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))   # when this quote was last seen
    yes_bid: Mapped[float | None] = mapped_column(Float)
    yes_ask: Mapped[float | None] = mapped_column(Float)
    mid: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(12))           # "poll" (our snapshots) | "candles" (Kalshi history)


# ----------------------------------------------------------------------------- flags & bets (Phase 4+)

class Flag(Base):
    """Every candidate with edge ≥ edge.log_min_edge, so thresholds and suppression windows can
    be tuned from data. `status`:
        flag             passes the thresholds, is the player's best rung, no suppression reason
        suppressed       would be a flag, but a suppression rule applies (see `reasons`)
        not_best         passes the thresholds but another rung for the same player is better
        below_threshold  edge between log_min_edge and the flag thresholds (logged for analysis)"""
    __tablename__ = "flags"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    kind: Mapped[str] = mapped_column(String(8))             # "pregame" | "live"
    status: Mapped[str] = mapped_column(String(12), index=True)
    reasons: Mapped[list] = mapped_column(JSON)              # [{"code": ..., "detail": ...}] ([] if flagged)
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    game_id: Mapped[str | None] = mapped_column(String(24), index=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"))
    ticker: Mapped[str] = mapped_column(ForeignKey("prop_markets.ticker"))
    stat: Mapped[str] = mapped_column(String(8))
    line: Mapped[float] = mapped_column(Float)
    side: Mapped[str] = mapped_column(String(3))              # "yes" | "no"
    model_prob: Mapped[float] = mapped_column(Float)
    ask: Mapped[float] = mapped_column(Float)
    fee_per_contract: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)
    ev_per_dollar: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(8))        # "normal" | "low"
    inputs: Mapped[dict] = mapped_column(JSON)                # everything needed to reproduce it
    result: Mapped[str | None] = mapped_column(String(8))     # win | loss | void (Phase 6)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)


class Bet(Base):
    __tablename__ = "bets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flag_id: Mapped[int | None] = mapped_column(ForeignKey("flags.id"))
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ticker: Mapped[str] = mapped_column(ForeignKey("prop_markets.ticker"))
    side: Mapped[str] = mapped_column(String(3))
    contracts: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float)
    result: Mapped[str | None] = mapped_column(String(8))   # win | loss | void
    pnl: Mapped[float | None] = mapped_column(Float)
