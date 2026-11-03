"""HttpClient behaviour with a fake session: no network, no real sleeping."""

import requests

from propedge.http import HttpClient


class FakeResponse:
    def __init__(self, status, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers = headers or {}
        self.url = "https://example.test/x"

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = 0
        self.headers = {}
        self.proxies = {}

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def make(queue, **kw):
    clock = FakeClock()
    settings = dict(min_interval_s=1.0, timeout_s=5, max_retries=2, backoff_s=2)
    settings.update(kw)
    client = HttpClient("test", sleep=clock.sleep, clock=clock.time, **settings)
    client.session = FakeSession(queue)
    return client, clock


def test_success_returns_data_and_timestamp():
    client, _ = make([FakeResponse(200, {"a": 1})])
    r = client.get_json("https://example.test/x")
    assert r.ok and r.data == {"a": 1} and r.status == 200
    assert r.fetched_at.tzinfo is not None


def test_retries_5xx_with_exponential_backoff():
    client, clock = make([FakeResponse(503), FakeResponse(502), FakeResponse(200, {"ok": True})])
    r = client.get_json("https://example.test/x")
    assert r.ok and r.attempts == 3
    # backoff 2s then 4s (rate-limit waits are 0 because the backoff already exceeds 1s)
    assert clock.sleeps == [2, 4]


def test_retries_timeouts_then_gives_up():
    client, _ = make([requests.exceptions.ReadTimeout("slow")] * 3)
    r = client.get_json("https://example.test/x")
    assert not r.ok and r.attempts == 3 and "ReadTimeout" in r.error


def test_403_is_not_retried():
    client, _ = make([FakeResponse(403, text="<HTML><TITLE>Access Denied</TITLE></HTML>")])
    r = client.get_json("https://example.test/x")
    assert not r.ok and r.attempts == 1 and client.session.calls == 1
    assert r.error == "HTTP 403: Access Denied"


def test_429_honours_retry_after():
    client, clock = make([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {})])
    assert client.get_json("https://example.test/x").ok
    assert clock.sleeps == [7]


def test_rate_limit_spaces_requests():
    client, clock = make([FakeResponse(200, {}), FakeResponse(200, {})], min_interval_s=1.5)
    client.get_json("https://example.test/x")
    client.get_json("https://example.test/x")
    assert clock.sleeps == [1.5]
