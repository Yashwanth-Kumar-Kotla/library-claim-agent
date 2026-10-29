import cv2
import numpy as np

from libraryclaim.fetch import Fetched


def textured(h=480, w=640, seed=0):
    """Busy, sharp image: random blocks plus text, roughly like a shelf of spines."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, np.uint8)
    for _ in range(300):
        x, y = rng.integers(0, w), rng.integers(0, h)
        color = tuple(int(c) for c in rng.integers(0, 230, 3))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 40)), y + int(rng.integers(5, 80))), color, -1)
    for i in range(12):
        cv2.putText(img, "SPINE TITLE", (10, 30 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return img


class FakeWeb:
    """Stands in for Fetcher. Routes by substring of the URL; records every call."""

    def __init__(self, routes=None, fail=()):
        self.routes = routes or {}
        self.fail = fail
        self.calls = []

    def __call__(self, url, params=None, headers=None):
        self.calls.append((url, params or {}, headers or {}))
        for bad in self.fail:
            if bad in url:
                from libraryclaim.fetch import FetchError
                raise FetchError(f"{url}: simulated outage")
        for needle, data in self.routes.items():
            if needle in url:
                payload = data(url, params or {}, headers or {}) if callable(data) else data
                return Fetched(payload, url, "2026-10-28T20:00:00+00:00")
        return Fetched({}, url, "2026-10-28T20:00:00+00:00")
