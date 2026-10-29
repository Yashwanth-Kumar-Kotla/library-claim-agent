"""Spine sizes on a shelf with no marker in view, from books whose real heights are known.

The depth model gives every spine a size, but its overall scale can be off by 10-25% per image.
Identified books have published dimensions (Google Books lists them for many printed editions).
One catalogue height cannot be trusted for one book (a hardcover and its paperback differ by 6 cm),
so the scale is a consensus: the single factor that makes the most books agree, within a few
percent, with one of their known edition heights. Misreads and wrong editions don't get a vote.

If too few books have known heights, the brief's fallback applies: standard book formats, here the
typical height of a shelf of mixed hardcovers and paperbacks, flagged as approximate.
"""

from __future__ import annotations

import re
import statistics

from rapidfuzz import fuzz

from .fetch import FetchError, GetJson
from .resolve import GOOGLE_BOOKS, main_title, norm, records

TOLERANCE = 0.04          # a book agrees with a scale if its scaled height is within 4% of an edition height
MIN_AGREEING = 3
TYPICAL_HEIGHT_CM = 21.5  # median of a shelf of mixed trade paperbacks (~21) and hardcovers (~24)


def parse_cm(text) -> float | None:
    m = re.match(r"\s*([\d.]+)\s*(cm|in|mm)?", str(text or ""))
    if not m:
        return None
    v = float(m.group(1)) * {"cm": 1, "in": 2.54, "mm": 0.1, None: 1}[m.group(2)]
    return v if 8 <= v <= 50 else None


def edition_heights(title: str, author: str, get_json: GetJson, key: str, limit: int = 3) -> list[float]:
    """Published heights (cm) of up to `limit` printed editions of this title and author."""
    if not key or not title:
        return []
    try:
        items = records(get_json(GOOGLE_BOOKS, {"q": f"{main_title(title)} {author}".strip(), "maxResults": 8,
                                                "printType": "books", "key": key}).data, "items")
    except FetchError:
        return []
    heights, checked = [], 0
    for it in items:
        v = it.get("volumeInfo") if isinstance(it.get("volumeInfo"), dict) else {}
        if fuzz.ratio(norm(main_title(v.get("title", ""))), norm(main_title(title))) < 90:
            continue
        checked += 1
        try:
            vol = get_json(f"{GOOGLE_BOOKS}/{it.get('id', '')}", {"key": key}).data
        except FetchError:
            continue
        dims = ((vol or {}).get("volumeInfo") or {}).get("dimensions") or {}
        h = parse_cm(dims.get("height"))
        if h:
            heights.append(round(h, 1))
        if checked >= limit:
            break
    return sorted(set(heights))


def consensus_scale(pairs: list[tuple[float, list[float]]]) -> tuple[float, int] | None:
    """pairs: (measured height, known edition heights). Returns (scale, books agreeing) or None."""
    candidates = sorted({h / raw for raw, hs in pairs if raw > 0 for h in hs})
    best: tuple[int, float] | None = None
    for k in candidates:
        agree = [k2 for raw, hs in pairs for k2 in [min((h / raw for h in hs), key=lambda r: abs(r - k), default=None)]
                 if k2 is not None and abs(k2 - k) / k <= TOLERANCE]
        if best is None or len(agree) > best[0]:
            best = (len(agree), statistics.median(agree))
    if best is None or best[0] < MIN_AGREEING:
        return None
    return best[1], best[0]


def format_scale(raw_heights: list[float]) -> float | None:
    """Fallback: assume the shelf's median book is a typical-height book."""
    raw = [h for h in raw_heights if h > 0]
    if len(raw) < 5:
        return None
    return TYPICAL_HEIGHT_CM / statistics.median(raw)
