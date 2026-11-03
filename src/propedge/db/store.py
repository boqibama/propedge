"""Write helpers shared by the historical backfill and the live poller: upserts and identity
resolution (mapping NBA / ESPN / Kalshi players and games onto our own rows)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from propedge.db.models import Base, Game, Player, Team
from propedge.ids import game_key, name_key, season_for

log = logging.getLogger(__name__)


def upsert(session: Session, model: type[Base], rows: Iterable[dict[str, Any]],
           conflict_cols: list[str], update_cols: list[str] | None = None) -> int:
    """INSERT ... ON CONFLICT DO UPDATE (works on SQLite and Postgres). Returns rows sent."""
    rows = list(rows)
    if not rows:
        return 0
    dialect = session.get_bind().dialect.name
    insert = {"sqlite": sqlite.insert, "postgresql": postgresql.insert}[dialect]
    # SQLite caps the number of bound parameters per statement, so send in chunks.
    chunk = max(1, 30000 // max(1, len(rows[0])))
    for i in range(0, len(rows), chunk):
        stmt = insert(model).values(rows[i:i + chunk])
        cols = update_cols if update_cols is not None else [c for c in rows[0] if c not in conflict_cols]
        if cols:
            stmt = stmt.on_conflict_do_update(index_elements=conflict_cols,
                                              set_={c: stmt.excluded[c] for c in cols})
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
        session.execute(stmt)
    return len(rows)


def ensure_game(session: Session, game_date: date, away: str, home: str, **fields: Any) -> str:
    """Create or update a game row; returns its id ('2026-10-21_GSW_LAL'). None-valued fields are ignored."""
    gid = game_key(game_date, away, home)
    game = session.get(Game, gid)
    if game is None:
        # Neutral-site games (NBA Cup, international) can be listed home/away the other way round
        # by another source: use the existing row and translate the fields to its orientation.
        swapped = session.get(Game, game_key(game_date, home, away))
        if swapped is not None:
            translate = {"away_score": "home_score", "home_score": "away_score"}
            fields = {translate.get(k, k): v for k, v in fields.items()}
            if fields.get("home_spread") is not None:
                fields["home_spread"] = -fields["home_spread"]
            game, gid = swapped, swapped.id
    if game is None:
        game = Game(id=gid, season=season_for(game_date), game_date=game_date, away=away, home=home)
        session.add(game)
    for k, v in fields.items():
        if v is not None:
            setattr(game, k, v)
    return gid


def ensure_team(session: Session, tricode: str, **fields: Any) -> None:
    team = session.get(Team, tricode)
    if team is None:
        team = Team(tricode=tricode)
        session.add(team)
    for k, v in fields.items():
        if v is not None:
            setattr(team, k, v)


class PlayerResolver:
    """Maps provider ids / names to `players.id`, creating players when a provider shows us
    someone new. Loads the whole (small) players table into memory once."""

    def __init__(self, session: Session):
        self.session = session
        self.by_nba: dict[int, Player] = {}
        self.by_espn: dict[str, Player] = {}
        self.by_key: dict[str, list[Player]] = {}
        for p in session.scalars(select(Player)):
            self._index(p)
        self.unmatched_names: set[tuple[str, str | None]] = set()

    def _index(self, p: Player) -> None:
        if p.nba_player_id is not None:
            self.by_nba[p.nba_player_id] = p
        if p.espn_athlete_id is not None:
            self.by_espn[p.espn_athlete_id] = p
        bucket = self.by_key.setdefault(p.name_key, [])
        if p not in bucket:
            bucket.append(p)

    def _new(self, **fields: Any) -> Player:
        p = Player(**fields)
        self.session.add(p)
        self.session.flush()          # assigns p.id
        self._index(p)
        return p

    def _match_name(self, name: str, team: str | None, free_attr: str) -> Player | None:
        """Unique player with this name (on this team if possible) whose `free_attr` id is unset."""
        candidates = [p for p in self.by_key.get(name_key(name), []) if getattr(p, free_attr) is None]
        on_team = [p for p in candidates if team and p.team == team]
        if len(on_team) == 1:
            return on_team[0]
        return candidates[0] if len(candidates) == 1 else None

    def from_nba(self, nba_id: int, name: str, team: str | None) -> int:
        p = self.by_nba.get(nba_id) or self._match_name(name, team, "nba_player_id")
        if p is None:
            p = self._new(nba_player_id=nba_id, name=name, name_key=name_key(name), team=team)
        else:
            p.nba_player_id = nba_id
            p.team = team or p.team
            self._index(p)
        return p.id

    def from_espn(self, espn_id: str, name: str, team: str | None, position: str | None = None) -> int:
        p = self.by_espn.get(espn_id) or self._match_name(name, team, "espn_athlete_id")
        if p is None:
            log.info("new player from ESPN (no NBA match yet): %s (%s)", name, team)
            p = self._new(espn_athlete_id=espn_id, name=name, name_key=name_key(name), team=team,
                          position=position)
        else:
            p.espn_athlete_id = espn_id
            p.team = team or p.team
            p.position = position or p.position
            self._index(p)
        return p.id

    def from_name(self, name: str, team: str | None) -> int | None:
        """Look up only (used for odds providers). Records misses for the status report."""
        candidates = self.by_key.get(name_key(name), [])
        on_team = [p for p in candidates if team and p.team == team]
        match = on_team[0] if len(on_team) == 1 else (candidates[0] if len(candidates) == 1 else None)
        if match is None:
            self.unmatched_names.add((name, team))
        return match.id if match else None


def utc(dt: datetime) -> datetime:
    """SQLite drops tzinfo on read; use this to compare stored times with aware datetimes."""
    from datetime import timezone
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
