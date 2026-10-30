"""Gemini calls for the three things we ask a vision model: where are the spines, what does
this spine say, and what non-book objects are in this frame.

The model is asked only for what is printed or visible. It never names a book that isn't
spelled out on the spine (resolve.py does that against a catalogue) and never prices anything.
"""

from __future__ import annotations

import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .boxes import Box, dedupe_reads, extract_json, fill_gaps, merge_tiles, parse_boxes, tiles

# One call finds and reads every spine in a tile. Asking for the text alongside each box makes
# the model walk the shelf spine by spine (better recall than boxes alone), and it is one call
# per tile instead of one per book: measured 9 s for a 30-book row vs ~100 s before.
SPINE_PROMPT = """This image shows part of one bookshelf row. Go along the shelf from left to right and list EVERY book,
including thin ones, books lying flat or stacked, and books cut off at the image edge.
For each book return {"box_2d": [ymin, xmin, ymax, xmax], "lying_flat": true|false,
"text": "<all spine text you can actually read, in order>", "title": "", "author": "", "publisher": "",
"legibility": 0.0-1.0, "looks_antique": true|false}. Coordinates are integers on a 0-1000 grid of this image;
box the whole spine. looks_antique is true only for visible signs of an old or collectible copy: cloth or
leather binding with gilt or blind-stamped lettering, raised bands, heavy age wear or fading.
Only transcribe printed text; leave title/author empty when not printed or not legible. Never fill them from
what the book probably is. Library call-number stickers are not titles. Do not list bookends, shelf edges or
decorations. Return [] if there are no books."""

READ_PROMPT = """This is one book spine, rotated upright. Transcribe only text you can actually read on it.
Return JSON: {"text": "<all legible text, top to bottom>", "title": "", "author": "", "publisher": "",
"legibility": <0.0-1.0, how much of the spine text you could read with confidence>}.
Leave a field empty if it is not printed or not legible. Never fill a field from what you think the
book probably is. If nothing is legible, return {"text": "", "legibility": 0}."""

ITEMS_PROMPT = """List the contents of this room that an insurance contents claim would list, other than books:
furniture, shelving units, lamps, electronics, appliances, framed art and portraits, rugs, and decor or
plants worth roughly 25 US dollars or more new.
Return a JSON list of {"box_2d": [ymin, xmin, ymax, xmax], "category": "<one of: shelving, furniture,
lighting, electronics, appliance, art, rug, decor, other>", "name": "<the basic kind of object in 1 or 2 plain
words, the same words whichever angle it is seen from: desk, office chair, keyboard, monitor, laptop, bookcase,
bed, mirror, floor lamp, framed picture>", "description": "<short
visible description, including material>", "brand_model": "<only if a brand or model name is legible, else
empty>", "is_art": true|false}. Coordinates are 0-1000.
Do not list: small accessories and consumables (cables, chargers, adapters, power strips, remotes, pens,
sticky notes, scissors, tape, mugs, bottles); boxes, bags and packaging; people or anything a person is
wearing or holding; clothing; food and drink; rubbish; documents; screen contents; the building and its
fittings (doors, windows, walls, floors, built-in cabinets and counters, built-in appliances, radiators,
switches, sockets); anything seen through a doorway in another room; objects mostly cut off by the edge of the photo (they are
listed from a view that shows them whole). List each physical object once, even if it is large."""
# Small things that slip past the prompt; listing them cost a price search each and buried the
# real items (seen: 56 "items" in one home sweep, including sticky notes, scissors and a retail box).
NOT_CONTENTS = re.compile(r"\b(cables?|cords?|chargers?|charging|adapters?|power (strip|bank)|surge|sticky notes?|"
                          r"post-?it|pens?|pencils?|scissors|tape|packaging|retail box|cardboard|shipping box|"
                          r"carton|(shopping|paper|plastic|tote) bags?|bottles?|mugs?|cups?|wrappers?|tissues?|receipts?|"
                          r"papers|paper towels?)\b", re.I)
PARTIAL = re.compile(r"\b(portion|partial(ly)?|part of|base of|edge of|corner of|section of)\b", re.I)

EMPTY_WORDS = {"", "unknown", "n/a", "na", "none", "null", "illegible", "unreadable", "-", "?"}


class Model(Protocol):
    def __call__(self, jpeg: bytes, prompt: str) -> str: ...


class Meter:
    """Tokens actually billed, per model, from each response's usage metadata."""

    def __init__(self, prices: dict[str, tuple[float, float]]):
        self.prices = prices  # model -> (input $/1M, output $/1M); output includes thinking
        self.lock = threading.Lock()
        self.tokens: dict[str, list[int]] = {}

    def add(self, model: str, tokens_in: int, tokens_out: int) -> None:
        with self.lock:
            t = self.tokens.setdefault(model, [0, 0, 0])
            t[0] += tokens_in
            t[1] += tokens_out
            t[2] += 1

    def cost(self) -> dict:
        with self.lock:
            per_model = {}
            for m, (i, o, n) in self.tokens.items():
                pin, pout = self.prices.get(m, (0.0, 0.0))
                per_model[m] = {"calls": n, "input_tokens": i, "output_tokens": o,
                                "usd": round(i * pin / 1e6 + o * pout / 1e6, 4), "priced": m in self.prices}
            return {"usd": round(sum(v["usd"] for v in per_model.values()), 4), "by_model": per_model}


def gemini_model(api_key: str, model: str, thinking: str = "low", meter: Meter | None = None,
                 timeout_s: float = 45.0) -> Model:
    """Thinking is kept low: on spine detection, default thinking took 14 s per call vs 6 s,
    for the same number of books found; high thinking was slower and truncated its output.
    It is also most of the bill, since thinking tokens are charged as output."""
    from google import genai
    from google.genai import types

    # A hard timeout: one slow call must not hold up the packet (seen: a 56 s escalation read).
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                         thinking_config=types.ThinkingConfig(thinking_level=thinking))

    def call(jpeg: bytes, prompt: str) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
            config=config,
        )
        u = resp.usage_metadata
        if meter and u:
            meter.add(model, u.prompt_token_count or 0, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
        return resp.text or ""

    return call


def cached(model: Model, cache_dir: Path, name: str) -> Model:
    """Replay identical requests from disk. Re-running a saved sweep while tuning costs nothing;
    only new images or changed prompts reach the API. Not for measuring variance or cost."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    lock = threading.Lock()

    def call(jpeg: bytes, prompt: str) -> str:
        key = hashlib.sha256(name.encode() + b"|" + prompt.encode() + b"|" + jpeg).hexdigest()[:32]
        # The same request twice in one run is deliberate (independent detection passes):
        # each occurrence gets its own entry, so replays stay independent too.
        with lock:
            n = seen[key] = seen.get(key, -1) + 1
        path = cache_dir / (f"{key}.txt" if n == 0 else f"{key}.{n}.txt")
        if path.is_file():
            return path.read_text()
        reply = model(jpeg, prompt)
        path.write_text(reply)
        return reply

    return call


def encode(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()


def detect_spines(model: Model, img: np.ndarray, tile_w: int | None = None, passes: int = 2) -> tuple[list[Box], int]:
    """Spine boxes, each carrying its read text in box.extra, across a whole shelf row image.
    Tiles are a little narrower than the row is tall, which keeps ~10 spines per call. Every
    tile is sent `passes` times at once (same latency as one) and the passes are unioned,
    because a single pass on a dense row is not repeatable. Any failed call raises."""
    h, w = img.shape[:2]
    tile_w = tile_w or max(400, int(0.8 * h))
    jobs = [(p, i, t) for p in range(passes) for i, t in enumerate(tiles(w, h, tile_w))]

    def run(job):
        p, i, (x0, y0, x1, y1) = job
        reply = model(encode(img[y0:y1, x0:x1]), SPINE_PROMPT)
        return p, parse_boxes(reply, x1 - x0, y1 - y0, offset=(x0, y0), tile=i)

    with ThreadPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(run, jobs))
    per_pass = [[] for _ in range(passes)]
    dropped = 0
    for p, (boxes, d) in results:
        per_pass[p] += boxes
        dropped += d
    merged = sorted((merge_tiles(b) for b in per_pass), key=len, reverse=True)
    out = merged[0]
    for extra in merged[1:]:
        out = fill_gaps(out, extra)
    return dedupe_reads(out), dropped


def clean(value) -> str:
    s = " ".join(str(value or "").split())
    return "" if s.lower() in EMPTY_WORDS else s


@dataclass
class SpineText:
    text: str
    title: str
    author: str
    publisher: str
    legibility: float

    @property
    def readable(self) -> bool:
        return bool(self.title or len(self.text) >= 4) and self.legibility > 0


def parse_read(reply: str) -> SpineText:
    try:
        data = extract_json(reply)
    except Exception:
        return SpineText("", "", "", "", 0.0)
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    return spine_text(data)


def spine_text(data) -> SpineText:
    """SpineText from a dict a model returned, either a single read or a detection entry."""
    if not isinstance(data, dict):
        return SpineText("", "", "", "", 0.0)
    try:
        legibility = float(data.get("legibility", 0) or 0)
    except (TypeError, ValueError):
        legibility = 0.0
    legibility = min(1.0, max(0.0, legibility if legibility <= 1 else legibility / 100))
    t = SpineText(clean(data.get("text")), clean(data.get("title")), clean(data.get("author")),
                  clean(data.get("publisher")), legibility)
    if not (t.text or t.title or t.author):
        t.legibility = 0.0  # a model that reports confidence about nothing is not confident
    return t


def spine_crop(img: np.ndarray, box: Box, pad: float = 0.08) -> np.ndarray:
    """Crop with a little margin and rotate so text runs horizontally for the reader.
    Upright spines usually read top-to-bottom, so rotate clockwise."""
    h, w = img.shape[:2]
    px, py = box.w * pad, box.h * pad
    x0, y0 = max(0, int(box.x0 - px)), max(0, int(box.y0 - py))
    x1, y1 = min(w, int(box.x1 + px)), min(h, int(box.y1 + py))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("box lies outside the image")
    upright = box.lying_flat is False or (box.lying_flat is None and box.h > box.w)
    return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE) if upright else crop


def read_spine(img: np.ndarray, box: Box, escalate: Model | None = None,
               escalate_below: float = 0.5) -> SpineText:
    """The text read during detection; if that was poor, one closer look at the cropped spine on
    the stronger model. The better read wins. Neither may invent text (see the prompts)."""
    first = spine_text(box.extra)
    if escalate is None or first.legibility >= escalate_below:
        return first
    second = parse_read(escalate(encode(spine_crop(img, box)), READ_PROMPT))
    return second if second.legibility > first.legibility else first


@dataclass
class SeenItem:
    box: Box
    category: str
    description: str
    brand_model: str
    is_art: bool
    name: str = ""      # short shop name, used to price it and to recognise it in the next view


CATEGORIES = {"shelving", "furniture", "lighting", "electronics", "appliance", "art", "rug", "decor", "other"}


def detect_items(model: Model, img: np.ndarray) -> list[SeenItem]:
    h, w = img.shape[:2]
    reply = model(encode(img), ITEMS_PROMPT)
    boxes, _ = parse_boxes(reply, w, h)
    out = []
    for b in boxes:
        category = clean(b.extra.get("category")).lower()
        if category not in CATEGORIES:
            category = "other"
        description, name, brand = (clean(b.extra.get(k)) for k in ("description", "name", "brand_model"))
        if category == "other" and "book" in description.lower():
            continue  # books are counted by the spine pipeline, not here
        if NOT_CONTENTS.search(f"{name} {description}") and not brand:
            continue
        if PARTIAL.search(description):
            continue  # a sliver of something; the view that shows it whole lists it
        if category == "other" and not brand:
            continue  # nothing insurable to say about an unnamed "other"
        is_art = b.extra.get("is_art") is True or category == "art"
        out.append(SeenItem(b, "art" if is_art else category, description, brand, is_art, name.lower()))
    return out
