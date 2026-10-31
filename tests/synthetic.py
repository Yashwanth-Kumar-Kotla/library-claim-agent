"""A synthetic sweep with known answers, and fakes that actually look at the pixels.

Shelf rows are drawn at 10 px/cm with a 15 cm ArUco marker. Each spine is a rectangle of a
unique hue, so the fake detector finds boxes by colour and the fake reader identifies a crop by
its colour. That way the real tiling, merging, cropping, rotation, scale and sizing code all run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from libraryclaim.vision import ITEMS_PROMPT

PX_PER_CM = 10


@dataclass
class Spine:
    title: str
    author: str
    h_cm: float
    t_cm: float
    flat: bool = False
    legible: bool = True
    extra_text: str = ""
    year: int = 1990
    new: tuple = (18, 19, 20, 21)
    used: tuple = (6, 7, 8, 9)
    looks_antique: bool = False
    publisher: str = ""


ROWS = {
    "A1": [
        Spine("Dune", "Frank Herbert", 17.8, 3.2, publisher="Ace"),
        Spine("The Hobbit", "J.R.R. Tolkien", 19.8, 2.2),
        Spine("Sapiens", "Yuval Noah Harari", 23.4, 3.5),
        Spine("Unreadable One", "", 21.0, 2.0, legible=False),
        Spine("Foundation", "", 18.0, 2.4),                                    # ambiguous: two authors
        Spine("Neuromancer", "William Gibson", 17.5, 2.0, extra_text="FIRST EDITION"),
        Spine("Middlemarch", "George Eliot", 20.0, 4.5, year=1871, looks_antique=True),  # old copy, by its look
        Spine("Great Expectations", "Charles Dickens", 19.0, 3.0, year=1861),  # modern copy of an old work: priced
    ],
    "A2": [
        Spine("Atlas Shrugged", "Ayn Rand", 23.0, 6.0, new=(400, 420, 450)),  # above threshold
        Spine("Obscure Pamphlet", "Nobody Known", 15.0, 0.6, new=(), used=()),  # no price anywhere
        Spine("Beloved", "Toni Morrison", 21.0, 2.6, flat=True),
        Spine("Normal People", "Sally Rooney", 19.8, 2.4, flat=True),
        Spine("Educated", "Tara Westover", 23.5, 3.0),
    ],
    "B1": [  # skipped by the policyholder
        Spine("Not Mine", "Someone", 20.0, 2.0),
    ],
}
ALL = [s for row in ROWS.values() for s in row]
HUES = {s.title: int(4 + i * 170 / len(ALL)) for i, s in enumerate(ALL)}


def color_for(title):
    hsv = np.uint8([[[HUES[title], 230, 210]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_row(spines, marker_id=0, with_marker=True):
    h, w = 420, 1900
    rng = np.random.default_rng(len(spines))
    img = (rng.normal(150, 6, (h, w, 3))).clip(0, 255).astype(np.uint8)  # low-saturation wood-ish texture
    if with_marker:
        m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, 150)
        img[250:400, 20:170] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        img[240:250, 10:180] = 255
        img[400:410, 10:180] = 255
        img[240:410, 10:20] = 255
        img[240:410, 170:180] = 255
    floor, x = 400, 230
    stack_y = floor
    truth = []
    for s in spines:
        hp, tp = int(s.h_cm * PX_PER_CM), int(s.t_cm * PX_PER_CM)
        if s.flat:
            y1, y0, x0, x1 = stack_y, stack_y - tp, x, x + hp
            stack_y = y0 - 4  # real stacks have a visible seam between books
        else:
            if stack_y != floor:
                x += int(max(sp.h_cm for sp in spines if sp.flat) * PX_PER_CM) + 4
                stack_y = floor
            x0, x1, y0, y1 = x, x + tp, floor - hp, floor
            x = x1 + 3
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color_for(s.title), -1)
        truth.append((s, (x0, y0, x1, y1)))
    return img, truth


def build_sweep(root: Path, rows=ROWS, skip=("B1",), room=True, items=True) -> dict:
    (root / "frames").mkdir(parents=True, exist_ok=True)
    rows_json, truth = [], {}
    for n, (shelf, spines) in enumerate(rows.items()):
        img, t = draw_row(spines, marker_id=n)
        ref = f"frames/{shelf}.jpg"
        cv2.imwrite(str(root / ref), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows_json.append({"shelf": shelf, "frames": [ref]})
        truth[shelf] = t
    (root / "rows.json").write_text(json.dumps(rows_json))
    (root / "sweep.json").write_text(json.dumps({"id": "syn-1", "captured_at": "2026-10-28T20:00:00Z", "device": "synthetic",
                                                 "duration_s": 150, "country": "US", "currency": "USD", "capture_path": "browser"}))
    (root / "assertions.json").write_text(json.dumps([{"kind": "skip_shelf", "shelf": s, "said": "skip that shelf, those are not mine"} for s in skip]))
    if items:
        room_img = np.full((600, 800, 3), 140, np.uint8)
        cv2.imwrite(str(root / "frames/room1.jpg"), room_img)
        cv2.imwrite(str(root / "frames/room2.jpg"), room_img)
        (root / "room_frames.json").write_text(json.dumps(["frames/room1.jpg", "frames/room2.jpg"]))
    if room:
        (root / "room.json").write_text(json.dumps({"corners": [[0, 0], [4.2, 0], [4.2, 3.1], [0, 3.1]], "height_m": 2.5,
                                                    "openings": [[0.9, 2.1]], "shelving": [[1.8, 2.0], [0.9, 2.0]],
                                                    "scale_method": "test fixture", "confidence": 0.9}))
    return truth


ITEMS_REPLY = json.dumps([
    {"box_2d": [100, 100, 600, 300], "category": "lighting", "description": "brass floor lamp"},
    {"box_2d": [200, 400, 500, 600], "category": "decor", "description": "framed portrait", "is_art": True},
    {"box_2d": [500, 600, 700, 900], "category": "appliance", "description": "espresso machine", "brand_model": "Breville BES870XL"},
    {"box_2d": [0, 0, 0, 0], "category": "rug"},  # junk box, dropped
])


def read_by_hue(hue: int) -> dict:
    s = min(ALL, key=lambda s: abs(HUES[s.title] - hue))
    if not s.legible:
        return {"text": "", "legibility": 0}
    return {"text": f"{s.title.upper()} {s.author.upper()} {s.extra_text}".strip(), "title": s.title,
            "author": s.author, "publisher": s.publisher, "legibility": 0.9, "looks_antique": s.looks_antique}


class FakeVision:
    """Detects spines by saturation, reads them by hue."""

    def __init__(self, miss_titles=()):
        self.miss = {HUES[t] for t in miss_titles}

    def __call__(self, jpeg: bytes, prompt: str) -> str:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if prompt.startswith("This image shows part of one bookshelf row"):
            mask = cv2.inRange(hsv, (0, 120, 60), (180, 255, 255))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            out = []
            for x, y, bw, bh, area in stats[1:]:
                if area < 40:
                    continue
                hue = int(np.median(hsv[y:y + bh, x:x + bw, 0]))
                if any(abs(hue - m) <= 1 for m in self.miss):
                    continue
                entry = {"box_2d": [y * 1000 / h, x * 1000 / w, (y + bh) * 1000 / h, (x + bw) * 1000 / w],
                         "lying_flat": bool(bw > bh)}
                entry.update(read_by_hue(hue))
                out.append(entry)
            return json.dumps(out)
        if prompt.startswith("This is one book spine"):
            cy, cx = h // 2, w // 2
            return json.dumps(read_by_hue(int(np.median(hsv[cy - 2:cy + 3, cx - 2:cx + 3, 0]))))
        if prompt == ITEMS_PROMPT:
            return ITEMS_REPLY
        return "[]"


def fake_catalogue():
    by_title = {s.title.lower(): s for s in ALL}

    def openlibrary(url, params, headers):
        q = params.get("q", "").lower()
        hits = [s for t, s in by_title.items() if q.startswith(t)]
        docs = [{"key": f"/works/{s.title.replace(' ', '')}", "title": s.title, "author_name": [s.author] if s.author else [],
                 "first_publish_year": s.year, "isbn": []} for s in hits]
        if q.startswith("foundation"):
            docs = [{"key": "/works/F1", "title": "Foundation", "author_name": ["Isaac Asimov"]},
                    {"key": "/works/F2", "title": "Foundation", "author_name": ["Mark Lawrence"]}]
        return {"docs": docs}

    def ebay(url, params, headers):
        q = params.get("q", "").lower()
        s = next((s for t, s in by_title.items() if q.startswith(t)), None)
        if "category_ids" not in params and s is None:  # item searches
            prices = (95, 100, 105) if "breville" in q else (40, 60, 80, 100)
            return {"itemSummaries": [{"title": q, "price": {"value": p, "currency": "USD"}} for p in prices]}
        if s is None:
            return {}
        prices = s.new if "NEW" in params["filter"] else s.used
        return {"itemSummaries": [{"title": f"{s.title} {s.author} paperback", "price": {"value": p, "currency": "USD"},
                                   "itemWebUrl": "https://www.ebay.com/itm/1"} for p in prices]}

    return {"openlibrary": openlibrary, "ebay": ebay}
