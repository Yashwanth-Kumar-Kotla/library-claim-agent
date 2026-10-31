"""Turning vision-model output into trustworthy boxes.

Models return almost-JSON: fenced, wrapped in an object, with boxes in [ymin, xmin, ymax, xmax]
on a 0-1000 grid, sometimes inverted or out of range. Everything is normalised here, and
anything that can't be repaired is dropped and counted rather than guessed at.

Dense shelves are detected in overlapping tiles (fewer objects per call = better recall),
so the same spine can come back twice. merge_tiles removes those duplicates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    lying_flat: bool | None = None
    score: float = 1.0
    extra: dict = field(default_factory=dict)
    tile: int = -1  # which detection call produced it; -1 = unknown

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


BOX_KEYS = ("box_2d", "box", "bbox", "bounding_box")


def box_value(item: dict):
    """The [ymin, xmin, ymax, xmax] list, under whatever key the model used. Asked for "box_2d",
    it has answered "box", "bbox" and, once, "box_2"."""
    for k in BOX_KEYS:
        if k in item:
            return item[k]
    for k, v in item.items():
        if k.lower().startswith(("box", "bbox", "bound")) and isinstance(v, list) and len(v) == 4:
            return v
    return None


def extract_json(text: str):
    """Pull the JSON value out of a model reply that may have fences or chatter around it."""
    if text is None:
        raise ValueError("empty model reply")
    s = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fenced:
        s = fenced.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (s.find("["), s.find("{")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in model reply")
    start = min(starts)
    end = max(s.rfind("]"), s.rfind("}"))
    if end <= start:
        raise ValueError("no JSON in model reply")
    return json.loads(s[start:end + 1])


def loose_objects(text: str) -> tuple[list[dict], int]:
    """Recover entries one by one from a list that doesn't parse as a whole. Seen live: one entry
    written as `"label": "lying_flat": false` made the whole tile (10+ books) unreadable.
    Returns (objects, entries that could not be recovered)."""
    objs, bad, depth, start, in_str, esc = [], 0, 0, None, False, False
    for i, ch in enumerate(text or ""):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                chunk = text[start:i + 1]
                for attempt in (chunk, re.sub(r'"[^"]*"\s*:\s*(?="[^"]*"\s*:)', "", chunk)):
                    try:
                        obj = json.loads(attempt)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        objs.append(obj)
                    break
                else:
                    bad += 1
    return objs, bad


def as_list(data, keys=("books", "spines", "items", "objects", "detections")) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in keys:
            if isinstance(data.get(k), list):
                return data[k]
    return []


def parse_boxes(text: str, width: int, height: int, offset: tuple[float, float] = (0, 0),
                min_px: float = 2.0, tile: int = -1) -> tuple[list[Box], int]:
    """Model reply -> pixel boxes in the full image. Returns (boxes, number dropped)."""
    dropped = 0
    try:
        raw = as_list(extract_json(text))
    except (ValueError, json.JSONDecodeError):
        raw, dropped = loose_objects(text)
    ox, oy = offset
    boxes = []
    for item in raw:
        b = box_value(item) if isinstance(item, dict) else None
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in b)
        except (TypeError, ValueError):
            dropped += 1
            continue
        ymin, ymax = sorted((ymin, ymax))
        xmin, xmax = sorted((xmin, xmax))
        clamp = lambda v: min(1000.0, max(0.0, v))
        x0, x1 = clamp(xmin) * width / 1000 + ox, clamp(xmax) * width / 1000 + ox
        y0, y1 = clamp(ymin) * height / 1000 + oy, clamp(ymax) * height / 1000 + oy
        if x1 - x0 < min_px or y1 - y0 < min_px:
            dropped += 1
            continue
        flat = item.get("lying_flat")
        if isinstance(flat, str):
            flat = {"true": True, "false": False, "yes": True, "no": False}.get(flat.lower())
        extra = {k: v for k, v in item.items()
                 if k not in ("label", "lying_flat") and not k.lower().startswith(("box", "bbox", "bound"))}
        boxes.append(Box(x0, y0, x1, y1, str(item.get("label", "")), flat if isinstance(flat, bool) else None,
                         extra=extra, tile=tile))
    return boxes, dropped


def tiles(width: int, height: int, tile_w: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Horizontal strips across a shelf panorama, each tile_w wide with the given overlap.
    The last tile is pinned to the right edge so nothing is missed."""
    if tile_w >= width:
        return [(0, 0, width, height)]
    step = max(1, int(tile_w * (1 - overlap)))
    xs = list(range(0, width - tile_w, step)) + [width - tile_w]
    return [(x, 0, x + tile_w, height) for x in dict.fromkeys(xs)]


def overlap_1d(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def legibility(b: Box) -> float:
    try:
        return float(b.extra.get("legibility") or 0)
    except (TypeError, ValueError):
        return 0.0


def text_key(b: Box) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(b.extra.get("text", "")).casefold())


def texts_differ(a: Box, b: Box) -> bool:
    """Both spines were read and the reads are clearly different books."""
    ta, tb = text_key(a), text_key(b)
    if len(ta) < 6 or len(tb) < 6 or ta in tb or tb in ta:
        return False  # a clipped spine shows part of the same text
    return fuzz.ratio(ta, tb) < 50


def same_object(a: Box, b: Box, min_cover: float = 0.6) -> bool:
    """Is b the same spine as a, seen again by an overlapping tile?

    Within one detection call the model lists each book once, so boxes from the same tile are
    never merged on geometry alone: leaning books have axis-aligned boxes that overlap their
    neighbours heavily (measured on a real bottom shelf: 21 books collapsed to 8 when they were).
    Across tiles, the smaller box being mostly inside the larger in both axes means a duplicate,
    which catches a spine cut in half at a tile edge, unless the two reads name different books."""
    ox = overlap_1d(a.x0, a.x1, b.x0, b.x1) / max(1e-6, min(a.w, b.w))
    oy = overlap_1d(a.y0, a.y1, b.y0, b.y1) / max(1e-6, min(a.h, b.h))
    same_tile = a.tile == b.tile and a.tile != -1
    ta, tb = text_key(a), text_key(b)
    if len(ta) >= 8 and ta == tb and ox >= (0.5 if same_tile else 0.3) and oy >= min_cover:
        return True  # the same long text twice on overlapping boxes: one book listed twice
    if same_tile:
        return False
    return ox >= min_cover and oy >= min_cover and not texts_differ(a, b)


def merge_tiles(boxes: list[Box]) -> list[Box]:
    """Collapse duplicates from overlapping tiles. The larger box wins the geometry: a box
    clipped by a tile edge is smaller than the same spine seen whole in the next tile.
    The more legible read wins the text, wherever it came from."""
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda b: b.area, reverse=True):
        twin = next((k for k in kept if same_object(b, k)), None)
        if twin is None:
            kept.append(b)
        elif legibility(b) > legibility(twin):
            twin.extra = dict(b.extra)
    return sorted(kept, key=lambda b: (b.x0, b.y0))


def fill_gaps(base: list[Box], other: list[Box]) -> list[Box]:
    """Union of two independent detection passes over the same row. Dense rows come back with a
    different subset of spines each time (measured: 26 then 12 books on the same row), so a
    second pass recovers what the first missed. Boxes are not merged on overlap (leaning
    neighbours overlap); a box from the other pass is added only if its centre falls in a gap
    that no base box covers."""
    out = list(base)
    for b in other:
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        covered = any(k.x0 <= cx <= k.x1 and k.y0 <= cy <= k.y1 for k in out)
        if not covered:
            out.append(b)
    return sorted(out, key=lambda b: (b.x0, b.y0))


def title_key(b: Box) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(b.extra.get("title", "")).casefold())


def dedupe_reads(boxes: list[Box], min_overlap: float = 0.3) -> list[Box]:
    """Last check on a row: two overlapping boxes that read as the same book are one book.
    Seen live after unioning passes: "Shiver" twice (25 px boxes overlapping by 23 px), "Days of
    Blood & Starlight" twice. Two real copies side by side only touch, so they stay apart."""
    out: list[Box] = []
    for b in sorted(boxes, key=lambda b: (b.x0, b.y0)):
        twin = None
        for k in out:
            ox = overlap_1d(k.x0, k.x1, b.x0, b.x1) / max(1e-6, min(k.w, b.w))
            if ox < min_overlap:
                continue
            tk, tb = title_key(k), title_key(b)
            xk, xb = text_key(k), text_key(b)
            if (len(tk) >= 5 and tk == tb) or (len(xk) >= 8 and xk == xb):
                twin = k
                break
        if twin is None:
            out.append(b)
        elif legibility(b) > legibility(twin):
            out[out.index(twin)] = b
    return out


def packed_spans(boxes: list[Box]) -> list[tuple[float, float, float, float]]:
    """Each upright spine's box with any overlap with its neighbours split down the middle. The
    model's boxes around thin or leaning spines are loose and overlap (seen: a 1.6 cm paperback
    boxed 11 cm wide); spines on a shelf cannot overlap, so the overlap is not part of either.
    Boxes that don't overlap, and flat books, keep their own width."""
    upright = sorted((i for i, b in enumerate(boxes) if not b.lying_flat and b.h >= b.w),
                     key=lambda i: boxes[i].x0 + boxes[i].x1)
    out = [b.as_tuple() for b in boxes]
    for k, i in enumerate(upright):
        b = boxes[i]
        x0, x1 = b.x0, b.x1
        if k > 0 and boxes[upright[k - 1]].x1 > x0:
            x0 = max(x0, (boxes[upright[k - 1]].x1 + b.x0) / 2)
        if k + 1 < len(upright) and boxes[upright[k + 1]].x0 < x1:
            x1 = min(x1, (b.x1 + boxes[upright[k + 1]].x0) / 2)
        if x1 > x0:
            out[i] = (x0, b.y0, x1, b.y1)
    return out
