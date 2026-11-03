"""Save raw provider payloads to disk (opt-in: `poll --record`) in the layout `replay` reads.

Size: roughly 3 MB per game-hour compressed, so a full 11-game night is ~150 MB. That's why
it's opt-in; the database already keeps everything the model needs.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

from propedge.ids import eastern_date


class RawRecorder:
    def __init__(self, root: Path, min_interval_s: dict[str, float] | None = None):
        self.root = root
        self.min_interval_s = min_interval_s or {}      # e.g. {"kalshi": 15}: fewer raw files than polls
        self._last: dict[tuple[str, str], datetime] = {}

    def __call__(self, kind: str, key: str, fetched_at: datetime, data: dict) -> None:
        gap = self.min_interval_s.get(kind)
        last = self._last.get((kind, key))
        if gap and last is not None and (fetched_at - last).total_seconds() < gap:
            return
        self._last[(kind, key)] = fetched_at
        folder = self.root / eastern_date(fetched_at).isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{kind}_{key}_{fetched_at.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
        with gzip.open(folder / name, "wt") as f:
            json.dump({"fetched_at": fetched_at.isoformat(), "data": data}, f)
