"""Replay a polled night from the database exactly as it was seen: every stored game snapshot, the box
score we had at that moment, and the plays first seen by then. Kalshi quotes come from the stored
~3 s snapshots (the live engine reads them as of each moment). Used to run new live logic (e.g. event
triggers) over nights recorded before it existed.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from propedge.data.live_base import LiveGameState, LivePlay, LivePlayerLine
from propedge.db import models as m
from propedge.db.store import utc


def states_from_db(session: Session, game_id: str) -> Iterator[tuple[m.LiveGameSnapshot, LiveGameState]]:
    game = session.get(m.Game, game_id)
    snaps = session.scalars(select(m.LiveGameSnapshot).where(
        m.LiveGameSnapshot.game_id == game_id, m.LiveGameSnapshot.provider == "espn", m.LiveGameSnapshot.is_mock.is_(False))
        .order_by(m.LiveGameSnapshot.fetched_at)).all()
    if not snaps or game is None:
        return
    players = {p.id: p for p in session.scalars(select(m.Player))}
    plays = session.scalars(select(m.Play).where(m.Play.game_id == game_id, m.Play.provider == "espn")
                            .order_by(m.Play.sequence)).all()
    box_snaps = sorted(set(session.scalars(select(m.LivePlayerSnapshot.snapshot_id).join(
        m.LiveGameSnapshot, m.LiveGameSnapshot.id == m.LivePlayerSnapshot.snapshot_id)
        .where(m.LiveGameSnapshot.game_id == game_id))))
    box_rows: dict[int, list] = {}
    for b in session.scalars(select(m.LivePlayerSnapshot).where(m.LivePlayerSnapshot.snapshot_id.in_(box_snaps))):
        box_rows.setdefault(b.snapshot_id, []).append(b)

    def espn(pid: int) -> str:
        p = players.get(pid)
        return p.espn_athlete_id if p and p.espn_athlete_id else f"x{pid}"

    current_box: list = []
    for snap in snaps:
        if snap.id in box_rows:
            current_box = box_rows[snap.id]
        at = utc(snap.fetched_at)
        lines = [LivePlayerLine(espn(b.player_id), players[b.player_id].name if b.player_id in players else "",
                                b.team, None, b.starter, bool(b.did_not_play), b.minutes, b.pts, b.fgm, b.fga, b.fg3m,
                                b.fg3a, b.ftm, b.fta, b.reb, b.ast, b.stl, b.blk, b.tov, b.pf, b.plus_minus)
                 for b in current_box]
        seen = [LivePlay(p.provider_play_id, p.sequence, p.period, p.clock_s, utc(p.wallclock) if p.wallclock else None,
                         p.play_type, p.text, p.team, p.scoring, p.score_value, p.away_score, p.home_score,
                         tuple(espn(i) for i in (p.player_ids or [])))
                for p in plays if utc(p.first_seen_at) <= at]
        yield snap, LiveGameState(
            provider="espn", provider_game_id=game.espn_event_id or game_id, fetched_at=at, game_date=game.game_date,
            away=game.away, home=game.home, status=snap.status, period=snap.period, clock_s=snap.clock_s,
            away_score=snap.away_score, home_score=snap.home_score, home_spread=game.home_spread, total=game.total,
            players=lines, plays=seen)
