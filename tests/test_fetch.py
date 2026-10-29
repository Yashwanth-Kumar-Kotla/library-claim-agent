import json
import threading
import time

import httpx
import pytest

from libraryclaim.fetch import PER_HOST, FetchError, Fetcher


def fetcher(handler, tmp_path=None):
    waits = []
    f = Fetcher(cache_dir=tmp_path, transport=httpx.MockTransport(handler), sleep=waits.append)
    return f, waits


def test_ok_is_cached_and_stamped(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={"docs": []})
    f, _ = fetcher(handler, tmp_path)
    a = f("https://openlibrary.org/search.json", {"q": "dune"})
    b = f("https://openlibrary.org/search.json", {"q": "dune"})
    assert len(calls) == 1 and a.retrieved_at == b.retrieved_at and "q=dune" in a.url
    assert calls[0].headers["User-Agent"].startswith("libraryclaim")


def test_real_429_is_retried_with_retry_after():
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"ok": True})
    f, waits = fetcher(handler)
    assert f("https://api.example.org/x").data == {"ok": True}
    assert waits == [2.0]


def test_gives_up_after_retries_and_raises_fetch_error():
    f, waits = fetcher(lambda req: httpx.Response(503))
    with pytest.raises(FetchError):
        f("https://api.example.org/x")
    assert len(waits) == 2


def test_timeouts_retried_then_reported():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)
    f, waits = fetcher(handler)
    with pytest.raises(FetchError):
        f("https://api.example.org/x")
    assert len(waits) == 2


def test_404_not_retried_and_failures_not_cached(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(404)
    f, waits = fetcher(handler, tmp_path)
    for _ in range(2):
        with pytest.raises(FetchError):
            f("https://api.example.org/x")
    assert len(calls) == 2 and waits == [] and not list(tmp_path.iterdir())


def test_huge_retry_after_is_capped():
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(429, headers={"Retry-After": "3600"}) if n["i"] == 1 else httpx.Response(200, json={})
    f, waits = fetcher(handler)
    f("https://api.example.org/x")
    assert waits == [5.0]


def test_non_json_body_is_a_fetch_error():
    f, _ = fetcher(lambda req: httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(FetchError):
        f("https://api.example.org/x")


def test_concurrency_capped_per_host():
    live, peak, lock = {"n": 0}, {"n": 0}, threading.Lock()

    def handler(req):
        with lock:
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
        time.sleep(0.02)
        with lock:
            live["n"] -= 1
        return httpx.Response(200, json={})
    f, _ = fetcher(handler)
    threads = [threading.Thread(target=f, args=(f"https://api.example.org/{i}",)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak["n"] <= PER_HOST


def test_open_library_is_paced():
    f, waits = fetcher(lambda req: httpx.Response(200, json={}))
    f.clock = lambda: 100.0  # frozen clock: every request after the first must wait its turn
    for i in range(4):
        f(f"https://openlibrary.org/search.json?q={i}")
    assert waits == pytest.approx([0.5, 1.0, 1.5])


def test_other_hosts_not_paced():
    f, waits = fetcher(lambda req: httpx.Response(200, json={}))
    f.clock = lambda: 100.0
    for i in range(4):
        f(f"https://api.example.org/{i}")
    assert waits == []
