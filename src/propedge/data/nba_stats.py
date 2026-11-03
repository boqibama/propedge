"""stats.nba.com access through nba_api, with its standard settings (default headers).

nba_api makes its own HTTP requests, so we wrap each call with:
- a rate limiter (config: sources.nba_stats.min_interval_s)
- retries with exponential backoff for timeouts / connection errors
- a JSON file cache under data/cache/nba_stats/ (finished games never change, so many
  responses can be cached forever)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import requests

from propedge.config import load_config, resolve_path
from propedge.http import RateLimiter

log = logging.getLogger(__name__)

FOREVER = None   # cache_ttl_s value meaning "never expires"


class NbaStatsError(RuntimeError):
    pass


class NbaStatsClient:
    def __init__(self, config: dict | None = None, cache_dir: Path | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        cfg = config or load_config()
        s = cfg["sources"]["nba_stats"]
        self.timeout_s = s["timeout_s"]
        self.max_retries = s["max_retries"]
        self.backoff_s = s["backoff_s"]
        self.default_ttl_s = s["cache_ttl_s"]
        self.limiter = RateLimiter(s["min_interval_s"], sleep=sleep)
        self._sleep = sleep
        self.cache_dir = cache_dir or resolve_path(cfg["paths"]["data_dir"]) / "cache" / "nba_stats"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def reachable(self, timeout_s: float = 15) -> bool:
        """One quick, uncached call with no retries. Cloud servers are often blocked or silently dropped by
        stats.nba.com; a full backfill there would sit through hours of timeouts."""
        from nba_api.stats.endpoints import CommonTeamYears
        try:
            self.limiter.wait()
            CommonTeamYears(timeout=timeout_s).get_dict()
            return True
        except Exception as exc:          # any failure means "don't rely on it tonight"
            log.warning("stats.nba.com not reachable (%s: %s)", type(exc).__name__, exc)
            return False

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_dir / f"{endpoint}_{digest}.json"

    def call(self, endpoint_cls: type, cache_ttl_s: float | None | str = "default",
             refresh: bool = False, **params: Any) -> dict:
        """Call an nba_api endpoint class (e.g. LeagueGameLog) and return its `get_dict()`."""
        ttl = self.default_ttl_s if cache_ttl_s == "default" else cache_ttl_s
        path = self._cache_path(endpoint_cls.__name__, params)
        if path.exists() and not refresh:
            age = time.time() - path.stat().st_mtime
            if ttl is FOREVER or age < ttl:
                return json.loads(path.read_text())

        attempt = 0
        while True:
            attempt += 1
            self.limiter.wait()
            try:
                data = endpoint_cls(**params, timeout=self.timeout_s).get_dict()
                path.write_text(json.dumps(data))
                return data
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                    json.JSONDecodeError, requests.exceptions.JSONDecodeError) as exc:
                if attempt > self.max_retries:
                    raise NbaStatsError(f"{endpoint_cls.__name__}{params}: {type(exc).__name__}: {exc}") from exc
                wait = self.backoff_s * 2 ** (attempt - 1)
                log.warning("%s failed (%s); retry %d in %.0fs", endpoint_cls.__name__,
                            type(exc).__name__, attempt, wait)
                self._sleep(wait)


def result_set(data: dict, index: int = 0) -> list[dict]:
    """Turn nba_api's {'resultSets': [{'headers': [...], 'rowSet': [[...]]}]} into a list of dicts."""
    rs = data["resultSets"][index]
    return [dict(zip(rs["headers"], row)) for row in rs["rowSet"]]
