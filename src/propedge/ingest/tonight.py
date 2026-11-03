"""`python -m propedge tonight`: the whole game night in one command.

1. backfill (yesterday's games, matchups, spreads, today's schedule)
2. wait until `tonight.start_before_first_tip_min` before the first tip-off
3. poll + record until every game today is final (pre-game Kalshi prices included)

On a Mac it also runs `caffeinate` so the computer doesn't sleep while it works
(the lid must stay open, or the Mac must be on power with an external display).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Callable

from propedge.data.live_base import ScheduledGame

log = logging.getLogger(__name__)


def plan_start(games: list[ScheduledGame], now: datetime, lead: timedelta) -> datetime | None:
    """When to start polling: `lead` before the first game that hasn't finished.
    Returns None if there's nothing left to watch today, or `now` if we should start immediately."""
    pending = [g for g in games if g.status != "post"]
    if not pending:
        return None
    if any(g.status == "in" for g in pending):
        return now
    tips = [g.start_time for g in pending if g.start_time is not None]
    if not tips:
        return now
    return max(now, min(tips) - lead)


def wait_until(target: datetime, now: Callable[[], datetime], sleep: Callable[[float], None],
               log_every: timedelta) -> None:
    next_log = now()
    while (remaining := target - now()) > timedelta(0):
        if now() >= next_log:
            log.info("tonight: waiting %s until polling starts at %s", str(remaining).split(".")[0],
                     target.astimezone().strftime("%H:%M %Z"))
            next_log = now() + log_every
        sleep(min(remaining.total_seconds(), 60))


def keep_awake() -> subprocess.Popen | None:
    """macOS: stop idle sleep for as long as this process lives."""
    if shutil.which("caffeinate") is None:
        return None
    # -i: no idle sleep; -s: no system sleep while on power. Closing the lid on battery still sleeps.
    return subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
