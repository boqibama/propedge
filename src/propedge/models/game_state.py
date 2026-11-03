"""Rebuild in-game player states from play-by-play. Pure, no I/O.

From a game's plays (in order) we track, for every player: points, personal fouls, minutes on the
floor, whether he's on the floor now, and when he last checked out. This gives the state at ANY
moment of a past game (for backtests), and on-floor status during a live game (the box score
doesn't say who is on the floor).

Who starts each period: the box score's starters for Q1. For later periods (play-by-play doesn't list
them) a player is on the floor at the start of a period if his first event in that period is anything
other than checking IN; gaps are filled with the previous period's closing lineup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from propedge.data.live_base import LivePlay

REG_PERIODS, PERIOD_MIN, OT_MIN = 4, 12.0, 5.0


def elapsed_minutes(period: int, clock_s: float | None) -> float:
    """Game minutes elapsed at (period, seconds left in the period). OT periods are 5 minutes."""
    clock_min = (clock_s or 0.0) / 60
    if period <= REG_PERIODS:
        return (period - 1) * PERIOD_MIN + (PERIOD_MIN - clock_min)
    return REG_PERIODS * PERIOD_MIN + (period - REG_PERIODS - 1) * OT_MIN + (OT_MIN - clock_min)


def period_end(period: int) -> float:
    return elapsed_minutes(period, 0.0)


def is_personal_foul(play_type: str) -> bool:
    t = play_type.lower()
    return "foul" in t and "turnover" not in t and "technical" not in t


@dataclass
class PlayerState:
    team: str | None = None
    pts: int = 0
    fouls: int = 0
    minutes: float = 0.0
    on_court: bool = False
    last_out: float | None = None        # game minute he last checked out
    last_in: float | None = None


@dataclass
class GameStateAt:
    elapsed: float
    period: int
    away_score: int
    home_score: int
    players: dict[str, PlayerState] = field(default_factory=dict)


def _period_starters(plays: list[LivePlay], period: int, team_of: dict[str, str]) -> dict[str, set[str]]:
    first_event: dict[str, str] = {}
    for p in plays:
        if p.period != period:
            continue
        ids = p.provider_player_ids
        if p.play_type == "Substitution" and len(ids) == 2:
            first_event.setdefault(ids[0], "in")
            first_event.setdefault(ids[1], "out")
        else:
            for pid in ids:
                first_event.setdefault(pid, "other")
    out: dict[str, set[str]] = {}
    for pid, ev in first_event.items():
        if ev != "in" and pid in team_of:
            out.setdefault(team_of[pid], set()).add(pid)
    return out


def replay_states(plays: list[LivePlay], starters: dict[str, set[str]], team_of: dict[str, str],
                  sample_at: list[float]) -> list[GameStateAt]:
    """States at each game minute in `sample_at` (ascending). `starters` = {team: {player ids}}
    for Q1; `team_of` = {player id: team}. Plays must be sorted by sequence."""
    plays = sorted(plays, key=lambda p: p.sequence)
    players: dict[str, PlayerState] = {pid: PlayerState(team=t) for pid, t in team_of.items()}
    on = {team: set(ids) for team, ids in starters.items()}
    for team, ids in on.items():
        for pid in ids:
            players.setdefault(pid, PlayerState(team=team)).on_court = True
            players[pid].last_in = 0.0
    clock, period, away, home = 0.0, 1, 0, 0
    samples = list(sample_at)
    out: list[GameStateAt] = []

    def advance(to: float) -> None:
        nonlocal clock
        if to <= clock:
            return
        for ids in on.values():
            for pid in ids:
                players[pid].minutes += to - clock
        clock = to

    def snapshot(at: float) -> None:
        advance(at)
        out.append(GameStateAt(at, period, away, home, {pid: PlayerState(**vars(ps)) for pid, ps in players.items()}))

    for p in plays:
        t = elapsed_minutes(p.period, p.clock_s)
        while samples and samples[0] < t:
            snapshot(samples.pop(0))
        if p.period != period:                      # new period: set its lineup
            advance(period_end(period))
            period = p.period
            inferred = _period_starters(plays, period, team_of)
            for team in set(on) | set(inferred):
                lineup = set(inferred.get(team, set()))
                if len(lineup) < 5:                 # fill from the previous period's closing lineup
                    lineup |= set(list(on.get(team, set()) - lineup)[: 5 - len(lineup)])
                for pid in on.get(team, set()) - lineup:
                    players[pid].on_court, players[pid].last_out = False, period_end(period - 1)
                for pid in lineup - on.get(team, set()):
                    players.setdefault(pid, PlayerState(team=team))
                    players[pid].on_court, players[pid].last_in = True, period_end(period - 1)
                on[team] = lineup
        advance(t)
        ids = p.provider_player_ids
        if p.play_type == "Substitution" and len(ids) == 2:
            pin, pout = ids
            team = team_of.get(pin) or team_of.get(pout) or p.team
            if team is not None:
                on.setdefault(team, set()).discard(pout)
                on[team].add(pin)
            players.setdefault(pin, PlayerState(team=team))
            players.setdefault(pout, PlayerState(team=team))
            players[pin].on_court, players[pin].last_in = True, t
            players[pout].on_court, players[pout].last_out = False, t
        elif p.scoring and ids:
            players.setdefault(ids[0], PlayerState(team=team_of.get(ids[0])))
            players[ids[0]].pts += p.score_value
        elif is_personal_foul(p.play_type) and ids:
            players.setdefault(ids[0], PlayerState(team=team_of.get(ids[0])))
            players[ids[0]].fouls += 1
        if p.away_score is not None:
            away, home = p.away_score, p.home_score
    while samples:
        snapshot(samples.pop(0))
    return out
