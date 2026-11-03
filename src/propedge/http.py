"""Shared HTTP client: browser-like headers, per-source rate limiting, retries with
exponential backoff, optional local caching, and a UTC timestamp on every response.

Every data source gets its own `HttpClient` (built from the `sources:` section of
config.yaml via `make_client`), so one slow or throttled source never affects another.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
import requests_cache
from nba_api.stats.library.http import STATS_HEADERS as NBA_STATS_HEADERS

from propedge.config import load_config, resolve_path

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)

# NBA's servers reject requests that don't look like they come from nba.com in a browser.
NBA_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

SOURCE_HEADERS: dict[str, dict[str, str]] = {
    # nba_api's own defaults. Phase 0 showed hand-rolled headers make stats.nba.com hang.
    "nba_stats": dict(NBA_STATS_HEADERS),
    "nba_live": NBA_HEADERS,
    "espn": {"User-Agent": BROWSER_UA, "Accept": "application/json"},
    "kalshi": {"Accept": "application/json"},
    "odds_api": {"Accept": "application/json"},
}

# Status codes worth retrying. 403 is deliberately absent: it means "blocked", and
# hammering a server that has blocked you only makes the block last longer.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class FetchResult:
    """One HTTP response plus the metadata we need for staleness checks and logging."""

    url: str
    status: int | None
    data: Any
    fetched_at: datetime
    elapsed_s: float
    from_cache: bool = False
    attempts: int = 1
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _snippet(body: str, limit: int = 120) -> str:
    """Short, single-line version of an error body (the <title> if it's an HTML page)."""
    title = re.search(r"<title>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
    text = title.group(1) if title else body
    return " ".join(text.split())[:limit]


class RateLimiter:
    """Thread-safe "at most one call every `min_interval_s` seconds" gate."""

    def __init__(self, min_interval_s: float, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last_at: float | None = None

    def wait(self) -> None:
        """Block until at least `min_interval_s` has passed since the previous call."""
        with self._lock:
            now = self._clock()
            if self._last_at is not None:
                pause = self.min_interval_s - (now - self._last_at)
                if pause > 0:
                    self._sleep(pause)
            self._last_at = self._clock()


class HttpClient:
    def __init__(
        self,
        name: str,
        *,
        min_interval_s: float,
        timeout_s: float,
        max_retries: int,
        backoff_s: float,
        cache_ttl_s: int = 0,
        headers: dict[str, str] | None = None,
        cache_path: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = name
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep
        self.rate_limiter = RateLimiter(min_interval_s, sleep=sleep, clock=clock)

        if cache_ttl_s > 0 and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.session: requests.Session = requests_cache.CachedSession(
                str(cache_path), backend="sqlite", expire_after=cache_ttl_s
            )
        else:
            self.session = requests.Session()
        self.session.headers.update(headers or {})

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> FetchResult:
        """GET a JSON endpoint. Never raises: failures come back with `error` set."""
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            self.rate_limiter.wait()
            status: int | None = None
            retry_after: float | None = None
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_s)
                status = resp.status_code
                if status == 200:
                    return FetchResult(
                        url=resp.url,
                        status=status,
                        data=resp.json(),
                        fetched_at=datetime.now(timezone.utc),
                        elapsed_s=time.perf_counter() - start,
                        from_cache=bool(getattr(resp, "from_cache", False)),
                        attempts=attempt,
                        headers=dict(resp.headers),
                    )
                error = f"HTTP {status}: {_snippet(resp.text)}"
                retryable = status in RETRYABLE_STATUS
                if status == 429 and resp.headers.get("Retry-After", "").isdigit():
                    retry_after = float(resp.headers["Retry-After"])
                headers = dict(resp.headers)
            except requests.exceptions.JSONDecodeError as exc:
                error, retryable, headers = f"Response was not JSON: {exc}", False, {}
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                error, retryable, headers = f"{type(exc).__name__}: {exc}", True, {}

            if not retryable or attempt > self.max_retries:
                return FetchResult(
                    url=url,
                    status=status,
                    data=None,
                    fetched_at=datetime.now(timezone.utc),
                    elapsed_s=time.perf_counter() - start,
                    attempts=attempt,
                    headers=headers,
                    error=error,
                )
            # Exponential backoff: backoff_s, 2×, 4×, ... (or what the server asked for).
            self._sleep(retry_after or self.backoff_s * 2 ** (attempt - 1))


def make_client(source: str, config: dict[str, Any] | None = None) -> HttpClient:
    """Build the HttpClient for one entry of the `sources:` section in config.yaml."""
    config = config or load_config()
    settings = config["sources"][source]
    return HttpClient(
        source,
        min_interval_s=settings["min_interval_s"],
        timeout_s=settings["timeout_s"],
        max_retries=settings["max_retries"],
        backoff_s=settings["backoff_s"],
        cache_ttl_s=settings["cache_ttl_s"],
        headers=SOURCE_HEADERS.get(source, {}),
        cache_path=resolve_path(f"{config['paths']['http_cache']}_{source}"),
    )
