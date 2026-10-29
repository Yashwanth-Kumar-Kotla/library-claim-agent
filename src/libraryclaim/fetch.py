"""HTTP GET with an on-disk cache. Every response keeps the URL and the time it was fetched,
so a price in the packet can always be traced to the exact response it came from."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlencode, urlsplit

import httpx


@dataclass
class Fetched:
    data: Any
    url: str
    retrieved_at: str


class GetJson(Protocol):
    def __call__(self, url: str, params: dict | None = None, headers: dict | None = None) -> Fetched: ...


class FetchError(Exception):
    pass


def full_url(url: str, params: dict | None) -> str:
    if not params:
        return url
    return f"{url}?{urlencode(sorted(params.items()))}"


USER_AGENT = "libraryclaim/0.1 (home contents claim prototype)"
PER_HOST = 4               # Open Library answered 429 with 16 parallel requests
RETRIES = (0.5, 1.5)       # backoff before each retry; then give up and say so
RETRY_STATUS = {429, 500, 502, 503, 504}
# Seconds between requests per host. Open Library asks unidentified clients for about 1-3/s
# and refused all connections after a 16-way burst.
MIN_INTERVAL = {"openlibrary.org": 0.5}


class Fetcher:
    def __init__(self, cache_dir: Path | None = None, timeout: float = 8.0,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep):
        self.cache_dir = cache_dir
        self.client = httpx.Client(timeout=timeout, follow_redirects=True, transport=transport,
                                   headers={"User-Agent": USER_AGENT})
        self.now = now
        self.sleep = sleep
        self.hosts: dict[str, threading.Semaphore] = {}
        self.next_at: dict[str, float] = {}
        self.hosts_lock = threading.Lock()
        self.clock = time.monotonic

    def slot(self, url: str) -> threading.Semaphore:
        host = urlsplit(url).netloc
        with self.hosts_lock:
            return self.hosts.setdefault(host, threading.Semaphore(PER_HOST))

    def pace(self, url: str) -> None:
        host = urlsplit(url).netloc
        gap = MIN_INTERVAL.get(host)
        if not gap:
            return
        with self.hosts_lock:
            now = self.clock()
            at = max(now, self.next_at.get(host, 0.0))
            self.next_at[host] = at + gap
        if at > now:
            self.sleep(at - now)

    def get(self, url: str, params: dict | None, headers: dict | None) -> httpx.Response:
        """GET with a per-host concurrency cap and short retries on rate limits, server errors
        and timeouts. Anything else fails at once: a 404 won't improve by asking again."""
        for attempt, wait in enumerate((*RETRIES, None)):
            try:
                self.pace(url)
                with self.slot(url):
                    resp = self.client.get(url, params=params, headers=headers)
                if resp.status_code not in RETRY_STATUS or wait is None:
                    return resp
                retry_after = resp.headers.get("Retry-After", "")
                wait = min(5.0, float(retry_after)) if retry_after.replace(".", "", 1).isdigit() else wait
            except httpx.TimeoutException:
                if wait is None:
                    raise
            self.sleep(wait)
        raise AssertionError("unreachable")

    def __call__(self, url: str, params: dict | None = None, headers: dict | None = None) -> Fetched:
        target = full_url(url, params)
        # Auth headers are deliberately not part of the key: same request, same answer.
        # The marketplace header changes the answer, so it is.
        market = (headers or {}).get("X-EBAY-C-MARKETPLACE-ID", "")
        key = hashlib.sha256(f"{target}|{market}".encode()).hexdigest()[:24]
        cached = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cached and cached.exists():
            hit = json.loads(cached.read_text())
            return Fetched(hit["data"], hit["url"], hit["retrieved_at"])
        try:
            resp = self.get(url, params, headers)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise FetchError(f"{target}: {exc}") from exc
        result = Fetched(data, target, self.now().isoformat(timespec="seconds"))
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(result.__dict__))
        return result
