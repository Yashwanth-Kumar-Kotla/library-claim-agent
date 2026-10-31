"""The claim pipeline, run while the sweep is still happening.

The live agent hands over each shelf row the moment the policyholder finishes it. That row is
stitched, measured, detected and read in the background, and every book found becomes its own
job (closer read if needed, catalogue match, appraisal rules, prices) that starts immediately.
Room frames are handled the same way. So by the time the sweep ends, nearly everything is done;
finish() only waits for the last row, applies anything the policyholder said late, measures the
room, and writes the packet. Time to packet is the time of that tail, not of the whole job.

    run = SweepRun(root, deps, settings, on_event=push_to_screen)
    run.add_row("A1", ["frames/0012.jpg", ...])      # as the agent closes each row
    run.add_room_frame("frames/0101.jpg")
    run.add_assertion(Assertion("skip_shelf", shelf="B2", said="those aren't mine"))
    packet = run.finish(room_geometry)

run_pipeline() replays a saved sweep through the same object, so batch and live share one path.
A saved sweep directory holds:

    sweep.json         id, captured_at, device, duration_s, country, currency, capture_path
    frames/*.jpg       full-resolution evidence frames
    rows.json          [{"shelf": "A1", "frames": ["frames/0012.jpg", ...]}]
    room_frames.json   ["frames/0101.jpg", ...]
    room.json          {"corners": [[x, y], ...], "height_m": .., "openings": [[w, h]], "shelving": [[w, h]],
                        "scale_method": "..", "confidence": ..}                   (optional)
    assertions.json    [{"kind": "skip_shelf", "shelf": "B2", "said": "..."}]    (optional)
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
from rapidfuzz import fuzz

from . import report
from .config import Locale, Settings
from .fetch import GetJson
from .packet import finalize
from .pricing import RANGE_NOTE, price_book, price_item
from .resolve import Resolution, identify
from .review import Assertion, appraisal_reasons, excluded, says_print
from .room import Opening, room_figures
from .room_scan import Depth, ScanFrame, front_size_cm, item_size_cm, measure_room
from .book_scale import consensus_scale, edition_heights, format_scale
from .scale import PLAUSIBLE_HEIGHT_CM, PLAUSIBLE_THICKNESS_CM, find_markers, plane_from_marker, shelf_run_cm, spine_size
from .schema import Book, BookPrice, ClaimPacket, Dimensions, Item, ItemPrice, ReviewEntry, Room, Sweep
from .segment import row_image
from .vision import Meter, Model, SeenItem, SpineText, detect_items, detect_spines, read_spine, spine_crop

WORKERS = 16  # all network-bound: model calls, catalogue and price lookups


@dataclass
class Deps:
    vision: Model
    get_json: GetJson
    ebay_token: str = ""
    escalate: Model | None = None
    google_books_key: str = ""
    meter: Meter | None = None  # real token usage and cost, when the models report it
    depth: Depth | None = None  # metric depth model: room size from the turn-around scan, item sizes


@dataclass
class BookJob:
    book: Book
    future: Future
    take: int = 0  # which capture of its row; a re-captured row replaces the earlier take
    text: SpineText | None = None
    resolution: Resolution | None = None
    raw_spine: tuple[float, float] | None = None  # height, thickness from the depth model, scale unknown (no marker)
    edition_heights: list[float] = field(default_factory=list)  # published heights of this title's editions


COLOURS = {"black", "white", "grey", "silver", "gold", "brass", "wood", "oak", "walnut", "brown", "beige", "cream",
           "blue", "navy", "teal", "green", "red", "pink", "orange", "yellow", "purple", "chrome", "glass"}
COLOUR_ALIASES = {"gray": "grey", "wooden": "wood"}


def colours(text: str) -> set[str]:
    words = (COLOUR_ALIASES.get(w, w) for w in re.findall(r"[a-z]+", text.lower()))
    return {w for w in words if w in COLOURS}


MAX_ITEM_SEARCHES = 25  # distinct item price searches per sweep; the library also needs ~150 for books

@dataclass
class ItemJob:
    item: Item
    seen: SeenItem
    future: Future
    last_view: int = 0  # index of the most recent room view it was seen in
    note: str = ""
    raw_size_cm: tuple[float, float] | None = None  # front face from depth, before marker scale correction


def load(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def read_image(root: Path, ref: str):
    img = cv2.imread(str(root / ref))
    if img is None:
        raise FileNotFoundError(ref)
    return img


@dataclass
class Stats:
    """Latency per stage (summed over jobs, plus the slowest single job) and model call counts."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    seconds: dict = field(default_factory=dict)
    slowest: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)

    def add(self, stage: str, dt: float) -> None:
        with self.lock:
            self.seconds[stage] = round(self.seconds.get(stage, 0) + dt, 3)
            self.slowest[stage] = round(max(self.slowest.get(stage, 0), dt), 3)

    def counted(self, name: str, model: Model | None) -> Model | None:
        if model is None:
            return None

        def call(jpeg, prompt):
            with self.lock:
                self.calls[name] = self.calls.get(name, 0) + 1
            return model(jpeg, prompt)
        return call


class SweepRun:
    def __init__(self, root: Path, deps: Deps, settings: Settings, locale: Locale | None = None,
                 on_event: Callable[[str, dict], None] | None = None):
        self.root = Path(root)
        self.out = self.root / "out"
        (self.out / "rows").mkdir(parents=True, exist_ok=True)
        (self.out / "crops").mkdir(parents=True, exist_ok=True)
        self.deps, self.settings = deps, settings
        self.locale = locale or settings.locale
        self.on_event = on_event or (lambda kind, data: None)
        self.stats = Stats()
        self.vision = self.stats.counted("vision", deps.vision)
        self.escalate = self.stats.counted("vision_escalation", deps.escalate)
        self.pool = ThreadPoolExecutor(WORKERS)
        self.room_pool = ThreadPoolExecutor(1)
        self.room_views = 0
        self.lock = threading.Lock()
        self.rows: list[tuple[str, Future]] = []
        self.takes: dict[str, int] = {}
        # Everything handed to this run, in order: written next to the frames so the same sweep can
        # be replayed through a newer pipeline (python -m libraryclaim run <sweep>).
        self.journal: dict[str, list] = {"rows": [], "room_frames": [], "assertions": []}
        self.marker_images: list = []     # shelf rows with a marker in view: calibrate the depth model's scale
        self.scan: Future | None = None
        self.scan_result = None
        self.remarks: list[dict] = []
        self.room_frames: list[Future] = []
        self.books: list[BookJob] = []
        self.items: list[ItemJob] = []
        self.item_queries: set[str] = set()   # distinct price searches spent on items this sweep
        self.review: list[ReviewEntry] = []
        self.said: list[Assertion] = []
        self.shelf_run_cm = 0.0
        self.measured_any = False
        self.row_info: list[dict] = []
        self.started = time.perf_counter()

    # ---- inputs, called while the sweep is in progress ----

    def add_assertion(self, a: Assertion) -> None:
        with self.lock:
            self.said.append(a)
            self.journal["assertions"].append(asdict(a))

    def add_row(self, shelf: str, frame_refs: list[str]) -> Future:
        """Hand over a finished row. Handing the same shelf over again (the agent asked for a
        re-capture) replaces the earlier take once both are done."""
        with self.lock:
            take = self.takes[shelf] = self.takes.get(shelf, -1) + 1
            self.journal["rows"].append({"shelf": shelf, "frames": list(frame_refs)})
        f = self.pool.submit(self._guard, f"row {shelf}", self._row, shelf, frame_refs, take)
        with self.lock:
            self.rows.append((shelf, f))
        return f

    def add_remark(self, kind: str, said: str, shelf: str = "", title: str = "", description: str = "") -> None:
        """Something the policyholder said about a book ("that Dune is a first edition") or an
        item ("the portrait is a print"). Which line it means is settled in finish(), once
        every line exists; if nothing matches it goes to the review queue, never to a guess."""
        with self.lock:
            self.remarks.append({"kind": kind, "said": said, "shelf": shelf, "title": title,
                                 "description": description, "at": time.perf_counter() - self.started})

    def add_room_frame(self, ref: str) -> Future:
        with self.lock:
            self.journal["room_frames"].append(ref)
        # Detection starts now, in parallel with other frames; the results are then taken one view at a
        # time, in capture order, because whether an object was "just seen" depends on it. Taking them
        # one model call at a time left a minute of room frames still queued when a live sweep ended.
        ahead = self.pool.submit(self._detect, ref)
        f = self.room_pool.submit(self._guard, f"room frame {ref}", self._room_frame, ref, ahead)
        with self.lock:
            self.room_frames.append(f)
        return f

    def add_room_frames(self, refs: list[str]) -> Future:
        """Several room frames handled in capture order in one job, so item ids and which
        frame each item is credited to don't depend on which model call returns first."""
        with self.lock:
            self.journal["room_frames"].extend(refs)

        ahead = [self.pool.submit(self._detect, ref) for ref in refs]

        def all_in_order():
            for ref, a in zip(refs, ahead):
                self._guard(f"room frame {ref}", self._room_frame, ref, a)
        f = self.room_pool.submit(all_in_order)
        with self.lock:
            self.room_frames.append(f)
        return f

    def add_room_scan(self, frames: list[ScanFrame]) -> Future:
        """The turn-around scan: room outline and height from metric depth plus phone orientation.
        Runs in the background as soon as the turn is done."""
        def job():
            if self.deps.depth is None:
                self.note("room", "no depth model configured; room size not measured")
                return
            t0 = time.perf_counter()
            with self.lock:
                calib_imgs = list(self.marker_images[:2])
            calibration = [(img, *self.deps.depth(img)) for img in calib_imgs]
            try:
                self.scan_result = measure_room(frames, self.deps.depth, self.settings.marker_size_cm / 100, calibration)
            except ValueError as exc:
                self.note("room", f"room scan unusable: {exc}")
            self.stats.add("room_scan", time.perf_counter() - t0)
        self.scan = self.pool.submit(self._guard, "room scan", job)
        return self.scan

    def note(self, ref_id: str, reason: str) -> None:
        with self.lock:
            self.review.append(ReviewEntry(ref_id=ref_id, reason=reason))

    def _guard(self, what: str, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - reported in the review queue, never silent
            ref = what.split(" ", 1)[1] if " " in what else what
            self.note(ref, f"{what} failed ({type(exc).__name__}: {exc})")

    # ---- shelf rows ----

    def _row(self, shelf: str, refs: list[str], take: int = 0) -> None:
        if any(a.kind == "skip_shelf" and a.shelf == shelf for a in self.said):
            self.note(shelf, "shelf skipped at the policyholder's request")
            return
        t0 = time.perf_counter()
        frames = []
        for ref in refs:
            try:
                frames.append((ref, read_image(self.root, ref)))
            except FileNotFoundError:
                self.note(shelf, f"frame {ref} missing")
        if not frames:
            self.note(shelf, "no usable frames for this shelf row")
            return
        ri = row_image(frames)
        ref = f"out/rows/{shelf}.jpg" if take == 0 else f"out/rows/{shelf}-take{take}.jpg"
        cv2.imwrite(str(self.root / ref), ri.image)
        if ri.note:
            self.note(shelf, ri.note)
        markers = find_markers(ri.image)
        if markers:
            with self.lock:
                self.marker_images.append(ri.image)
        scale = plane_from_marker(max(markers, key=lambda m: m.side_px), self.settings.marker_size_cm) if markers else None
        if scale is None:
            if self.deps.depth is None:
                self.note(shelf, "no scale marker visible in this row; spine sizes not measured")
        elif not scale.reliable:
            self.note(shelf, "scale marker too small in frame; spine sizes approximate")
        self.stats.add("segment", time.perf_counter() - t0)

        t0 = time.perf_counter()
        try:
            boxes, dropped = detect_spines(self.vision, ri.image)
        except Exception as exc:  # noqa: BLE001
            self.note(shelf, f"spine detection failed ({type(exc).__name__}: {exc}); row not counted")
            return
        if dropped:
            self.note(shelf, f"{dropped} detections unusable and ignored")
        self.stats.add("detect_read", time.perf_counter() - t0)

        with self.lock:
            if scale is not None and boxes:
                self.shelf_run_cm += shelf_run_cm(scale, [b.as_tuple() for b in boxes])
                self.measured_any = True
            self.row_info.append({"shelf": shelf, "ref": ref, "method": ri.method, "books": len(boxes),
                                  "scale": scale.method if scale else None,
                                  "boxes": [b.as_tuple() for b in boxes]})
        self.on_event("row_detected", {"shelf": shelf, "books": len(boxes)})

        depth = None
        if scale is None and boxes and self.deps.depth is not None:
            t1 = time.perf_counter()
            depth = self.deps.depth(ri.image)   # no marker: measure now, settle the scale from known books later
            self.stats.add("depth", time.perf_counter() - t1)
        for pos, box in enumerate(boxes, start=1):
            bid = f"{shelf}-{pos:02d}" if take == 0 else f"{shelf}-{pos:02d}t{take}"
            b = Book(id=bid, shelf=shelf, position=pos,
                     frame_ref=f"{ref}#xyxy={int(box.x0)},{int(box.y0)},{int(box.x1)},{int(box.y1)}")
            if scale is not None:
                size = spine_size(scale, box.as_tuple(), box.lying_flat)
                if size.plausible:
                    b.spine_height_cm, b.spine_thickness_cm = size.height_cm, size.thickness_cm
                else:
                    self.note(bid, f"implausible spine size {size.height_cm} x {size.thickness_cm} cm; not reported")
            job = BookJob(b, Future(), take)
            if depth is not None:
                face = front_size_cm(box.as_tuple(), depth[0], depth[1])
                if face:
                    flat = box.lying_flat if box.lying_flat is not None else face[0] > face[1]
                    job.raw_spine = (face[0], face[1]) if flat else (face[1], face[0])
            with self.lock:
                self.books.append(job)
            job.future = self.pool.submit(self._guard, f"book {bid}", self._book, job, ri.image, box)

    # ---- one job per book, started as soon as its row is read ----

    def _book(self, job: BookJob, img, box) -> None:
        b = job.book
        t0 = time.perf_counter()
        cv2.imwrite(str(self.out / "crops" / f"{b.id}.jpg"), spine_crop(img, box))
        try:
            job.text = read_spine(img, box, escalate=self.escalate)
        except Exception as exc:  # noqa: BLE001
            self.note(b.id, f"spine could not be read ({type(exc).__name__}: {exc})")
            job.text = read_spine(img, box)  # fall back to the read from detection
        if not job.text.readable:
            self.note(b.id, "spine not legible; logged as unidentified with its dimensions")
            self.on_event("book", b.model_dump())
            return
        t1 = time.perf_counter()
        self.stats.add("read", t1 - t0)

        res, notes = identify(job.text, self.deps.get_json, self.deps.google_books_key)
        job.resolution = res
        b.id_confidence = res.confidence
        t2 = time.perf_counter()
        self.stats.add("resolve", t2 - t1)
        if res.status != "identified":
            self.note(b.id, res.reason + (f" ({'; '.join(notes)})" if notes else ""))
            self.on_event("book", b.model_dump())
            return
        c = res.candidate
        b.status, b.title, b.author = "identified", c.title, ", ".join(c.authors)
        b.edition, b.isbn, b.publisher = res.edition, res.isbn, job.text.publisher
        self.on_event("book", b.model_dump())
        if job.raw_spine and self.deps.google_books_key:
            job.edition_heights = edition_heights(c.title, c.authors[0] if c.authors else "", self.deps.get_json,
                                                  self.deps.google_books_key)

        # Rare books are flagged before any price lookup; a price at or over the threshold flags after.
        reasons = appraisal_reasons(spine_text=job.text.text, year=c.year if res.edition else None,
                                    looks_antique=box.extra.get("looks_antique") is True,
                                    threshold=self.settings.appraisal_threshold)
        if not reasons:
            prices = price_book(self.deps.get_json, self.deps.ebay_token, self.locale, b.title, b.author.split(",")[0], b.isbn)
            reasons = appraisal_reasons(replacement=prices.replacement.amount if prices.replacement else None,
                                        used=prices.used.amount if prices.used else None,
                                        threshold=self.settings.appraisal_threshold)
            if not reasons:
                for name, q in (("replacement_cost", prices.replacement), ("used_value", prices.used)):
                    if q:
                        setattr(b, name, BookPrice(amount=q.amount, source=q.source, url=q.url, retrieved_at=q.retrieved_at,
                                                   converted=q.converted, condition_assumed=q.condition))
                text = "; ".join(n for n in prices.notes if n)
                if prices.replacement is None:
                    self.note(b.id, "no replacement price found; excluded from totals: " + text)
                elif text:
                    self.note(b.id, text)
        if reasons:
            b.status = "needs_appraisal"
            self.note(b.id, "needs human appraisal: " + "; ".join(reasons))
        self.stats.add("price", time.perf_counter() - t2)
        self.on_event("book", b.model_dump())

    # ---- non-book items: detected per frame, priced as soon as a new one appears ----

    def _detect(self, ref: str):
        t0 = time.perf_counter()
        try:
            img = read_image(self.root, ref)
        except FileNotFoundError:
            return None, []
        seen = detect_items(self.vision, img)
        self.stats.add("items_detect", time.perf_counter() - t0)
        return img, seen

    def _room_frame(self, ref: str, ahead: Future | None = None) -> None:
        img, seen = ahead.result() if ahead is not None else self._detect(ref)
        if img is None:
            self.note(ref, "room frame missing")
            return
        with self.lock:
            self.room_views += 1
            view = self.room_views
            fresh = []
            for s in seen:
                twin = self._same_item(s, view)
                if twin is not None:
                    twin.last_view = view
                else:
                    fresh.append(s)  # two sightings in one photo are two objects
        if not fresh:
            return  # nothing new in this view: no depth map needed (it is the slowest step)
        depth = None
        if self.deps.depth is not None:
            t1 = time.perf_counter()
            depth = self.deps.depth(img)
            self.stats.add("depth", time.perf_counter() - t1)
        for s in fresh:
            with self.lock:
                iid = f"I-{len(self.items) + 1:02d}"
                it = Item(id=iid, category=s.category, description=s.description, brand_model=s.brand_model,
                          frame_ref=f"{ref}#xyxy={int(s.box.x0)},{int(s.box.y0)},{int(s.box.x1)},{int(s.box.y1)}",
                          dimensions_cm=Dimensions(), confidence=0.6 if s.brand_model else 0.4)
                job = ItemJob(it, s, Future(), view)
                if depth is not None:
                    job.raw_size_cm = item_size_cm(s.box.as_tuple(), depth[0], depth[1])
                self.items.append(job)
            job.future = self.pool.submit(self._guard, f"item {iid}", self._price_item, job)
            self.on_event("item", it.model_dump())

    def _same_item(self, s: SeenItem, view: int) -> ItemJob | None:
        """Is this sighting an object we already have? The same object turns up again in the next
        view or two of a pan, described in different words ("black gaming chair", "black ergonomic
        office chair"), so recent sightings of the same category match loosely. Anything older must
        match closely: two similar lamps on opposite walls are two lamps."""
        text = f"{s.description} {s.brand_model}".lower()
        best, best_score = None, 0.0
        for j in self.items:
            if s.name and s.name == j.seen.name and s.brand_model == j.seen.brand_model:
                if view - j.last_view <= 3:
                    return j  # "office chair" again a moment later, whatever the wording or category
                if colours(s.description) & colours(j.seen.description) or not (colours(s.description) and colours(j.seen.description)):
                    return j  # seen again from across the room (the turn sees everything twice); a black
                              # and a brass "floor lamp" stay two lamps
            if j.seen.category != s.category:
                continue
            score = fuzz.token_set_ratio(text, f"{j.seen.description} {j.seen.brand_model}".lower())
            need = 55 if view - j.last_view <= 2 else 85
            if score >= need and score > best_score:
                best, best_score = j, score
        return best

    def _price_item(self, job: ItemJob, user_says_print: bool = False) -> None:
        t0 = time.perf_counter()
        s, it = job.seen, job.item
        query = (s.brand_model or s.name or s.description).lower()
        with self.lock:
            over = query not in self.item_queries and len(self.item_queries) >= MAX_ITEM_SEARCHES
            if not over:
                self.item_queries.add(query)  # a repeated query is answered from the cache, free
        if over:
            it.status, it.replacement_cost = "range", ItemPrice()
            job.note = f"not priced: this sweep reached its {MAX_ITEM_SEARCHES} item price searches"
            return
        p = price_item(self.deps.get_json, self.deps.ebay_token, self.locale, s.category, s.description, s.brand_model,
                       is_art=s.is_art, user_says_print=user_says_print, name=s.name)
        it.status = p.status
        it.replacement_cost = ItemPrice()
        if p.quote:
            it.replacement_cost = ItemPrice(low=p.quote.low, high=p.quote.high, source=p.quote.source,
                                            url=p.quote.url, retrieved_at=p.quote.retrieved_at)
        job.note = p.note if (p.status == "needs_appraisal" or p.quote is None or p.note) else ""
        self.stats.add("items_price", time.perf_counter() - t0)

    # ---- the tail: everything left once the policyholder stops ----

    def finish(self, room_geometry: dict | None = None, meta: dict | None = None) -> ClaimPacket:
        t_end = time.perf_counter()
        for _, f in list(self.rows):
            f.result()
        for f in list(self.room_frames):
            f.result()
        for job in list(self.books):
            job.future.result()
        for job in list(self.items):
            job.future.result()
        if self.scan is not None:
            self.scan.result()
        # A re-captured row replaces the earlier take, which may still have finished last.
        latest = dict(self.takes)
        stale = [j for j in self.books if j.take != latest.get(j.book.shelf, 0)]
        self.books = [j for j in self.books if j.take == latest.get(j.book.shelf, 0)]
        if stale:
            for shelf in sorted({j.book.shelf for j in stale}):
                self.note(shelf, f"row re-captured; earlier take of {shelf} discarded")
        self._scale_unmarked_rows()
        self.said += self._resolve_remarks()
        said = list(self.said)

        # A print stated after the item was priced as art: price it as a print now.
        for job in self.items:
            if job.seen.is_art and says_print(job.item.id, said):
                self._price_item(job, user_says_print=True)

        books = []
        for job in self.books:
            b = job.book
            gone = excluded(b.id, b.shelf, said)
            if gone:
                self.note(b.id, f"removed: policyholder said '{gone.said or gone.kind}'")
                continue
            flags = appraisal_reasons(said=[a for a in said if a.applies_to(b.id, b.shelf)],
                                      threshold=self.settings.appraisal_threshold)
            if flags and b.status == "identified":
                b.status = "needs_appraisal"
                self.note(b.id, "needs human appraisal: " + "; ".join(flags))
            books.append(b)
        books.sort(key=lambda b: (b.shelf, b.position))
        books = self._drop_double_sightings(books)

        items = []
        # Notes that apply to most items are listed once with their ids, not once per item (seen: 113
        # review lines for one room, nearly all the same two sentences, read out by the agent).
        common = {RANGE_NOTE: [], "approximate front-face size from monocular depth; depth not measured": [],
                  "dimensions not measured": []}
        for job in self.items:
            gone = excluded(job.item.id, "", said)
            if gone:
                self.note(job.item.id, f"removed: policyholder said '{gone.said or gone.kind}'")
                continue
            if job.note in common:
                common[job.note].append(job.item.id)
            elif job.note:
                self.note(job.item.id, job.note)
            scale = self.scan_result.scale_factor if self.scan_result else 1.0
            if job.raw_size_cm:
                w, h = job.raw_size_cm
                job.item.dimensions_cm = Dimensions(w=round(w * scale), h=round(h * scale))
                common["approximate front-face size from monocular depth; depth not measured"].append(job.item.id)
            else:
                common["dimensions not measured"].append(job.item.id)
            items.append(job.item)
        for reason, ids in common.items():
            if ids:
                self.note("items", f"{reason}: {', '.join(ids)}")

        if room_geometry is None and self.scan_result is not None:
            shelving = [(j.item.dimensions_cm.w / 100, j.item.dimensions_cm.h / 100) for j in self.items
                        if j.item.category == "shelving" and j.item.dimensions_cm.w and j.item in items]
            room_geometry = self.scan_result.as_room_json(shelving)
            for n in self.scan_result.notes:
                self.note("room", n)
            if room_geometry["height_m"] is None:
                room_geometry = None
                self.note("room", "room height not measured, so wall areas cannot be computed; floor outline kept in out/room_scan.json")
            (self.out / "room_scan.json").write_text(json.dumps(self.scan_result.as_room_json(shelving), indent=2))
        room = self._room(room_geometry)
        sweep = Sweep(**{k: v for k, v in (meta or {}).items() if k in Sweep.model_fields})
        sweep.country, sweep.currency = self.locale.country, self.locale.currency
        # Jobs finish in any order; the packet must not depend on thread timing.
        review = sorted(self.review, key=lambda r: (r.ref_id, r.reason))
        packet = ClaimPacket(sweep=sweep, room=room, books=books, items=items, review_queue=review)
        packet = finalize(packet, self.root, round(self.shelf_run_cm / 100, 2) if self.measured_any else None)
        (self.root / "claim_packet.json").write_text(packet.model_dump_json(indent=2))
        report.write(packet, self.root)
        self.pool.shutdown(wait=False)

        self.stats.seconds["after_sweep"] = round(time.perf_counter() - t_end, 3)
        self.stats.seconds["total"] = round(time.perf_counter() - self.started, 3)
        self._save_stage_outputs()
        return packet

    def _scale_unmarked_rows(self) -> None:
        """Spine sizes for shelves with no marker in view: the depth model's measurements, scaled by
        the consensus of identified books with known edition heights, else the standard-format
        fallback. Always per shelf: on real phone shelves the depth model read one shelf ~40% tall
        and another ~25% short, so one shelf's scale says nothing about the next."""
        rows: dict[str, list[BookJob]] = {}
        for j in self.books:
            if j.raw_spine:
                rows.setdefault(j.book.shelf, []).append(j)
        if not rows:
            return
        for shelf, jobs in rows.items():
            own = consensus_scale([(j.raw_spine[0], j.edition_heights) for j in jobs if j.edition_heights])
            if own:
                k, why = own[0], f"scaled to {own[1]} identified books of known height on this shelf"
            else:
                k = format_scale([j.raw_spine[0] for j in jobs])
                why = "scaled to a typical book height (standard formats); approximate"
            if k is None:
                self.note(shelf, "no marker and too few books to set a scale; spine sizes not reported")
                continue
            self.note(shelf, f"no marker in view: spine sizes from the depth model, {why}")
            run = 0.0
            for j in jobs:
                h, t = j.raw_spine[0] * k, j.raw_spine[1] * k
                if PLAUSIBLE_HEIGHT_CM[0] <= h <= PLAUSIBLE_HEIGHT_CM[1] and PLAUSIBLE_THICKNESS_CM[0] <= t <= PLAUSIBLE_THICKNESS_CM[1]:
                    j.book.spine_height_cm, j.book.spine_thickness_cm = round(h, 1), round(t, 2)
                    run += t
            # Without a marker there is no shelf plane to project onto; the books' own thicknesses
            # still give the run they occupy (a flat stack is over-counted, so it stays approximate).
            if run:
                self.shelf_run_cm += run
                self.measured_any = True
                self.note(shelf, "shelf run from the sum of spine thicknesses (no marker); approximate")

    def _resolve_remarks(self) -> list[Assertion]:
        """Tie each spoken remark to one line: a book by title on the shelf that was in view,
        an artwork by description. Ambiguous or unmatched remarks are reported, not applied."""
        out = []
        for r in self.remarks:
            if r["kind"] in ("is_print", "original") or (r["kind"] == "not_mine" and r["description"]):
                art_only = r["kind"] != "not_mine"
                cands = [j.item for j in self.items if j.seen.is_art or not art_only]
                pool = [(fuzz.token_set_ratio(r["description"].lower(), i.description.lower()), i) for i in cands]
                if art_only and len(cands) == 1 and not r["description"]:
                    pool = [(100, cands[0])]
            else:
                books = [j.book for j in self.books if (not r["shelf"] or j.book.shelf == r["shelf"]) and j.book.title]
                pool = [(fuzz.token_set_ratio(r["title"].lower(), b.title.lower()), b) for b in books] if r["title"] else []
            pool.sort(key=lambda x: x[0], reverse=True)
            ok = pool and pool[0][0] >= 70 and (len(pool) == 1 or pool[0][0] - pool[1][0] >= 10)
            if ok:
                out.append(Assertion(r["kind"], target=pool[0][1].id, said=r["said"], at_s=r["at"]))
            else:
                where = r["shelf"] or "the room"
                self.note(r["shelf"] or "room", f"policyholder said \"{r['said']}\" about '{r['title'] or r['description']}' "
                                                f"on {where}; could not tie it to one line, please check")
        return out

    def _drop_double_sightings(self, books: list[Book]) -> list[Book]:
        """Two overlapping boxes on one row that resolved to the same book are one book seen
        twice (leaning spines, or both detection passes). Real duplicate copies stand side by
        side and only touch, so they are kept. Only identified books are compared: the match
        to a catalogue record is the evidence, not the box."""
        def box(b):
            return [float(v) for v in b.frame_ref.split("#xyxy=")[1].split(",")]
        kept: list[Book] = []
        for b in books:
            twin = next((k for k in kept if k.shelf == b.shelf and k.status != "unidentified" and b.status != "unidentified"
                         and k.title == b.title and k.author == b.author and self._overlap(box(k), box(b)) >= 0.3), None)
            if twin is None:
                kept.append(b)
                continue
            keep, drop = (b, twin) if b.id_confidence > twin.id_confidence else (twin, b)
            kept[kept.index(twin)] = keep
            self.note(drop.id, f"same book as {keep.id} seen twice (overlapping boxes); removed from the count")
        return kept

    @staticmethod
    def _overlap(a, b) -> float:
        inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        return inter / max(1e-6, min(a[2] - a[0], b[2] - b[0]))

    def _room(self, geo: dict | None) -> Room:
        if not geo:
            self.note("room", "no room geometry captured")
            return Room()
        try:
            fig = room_figures(geo["corners"], geo["height_m"], [Opening(*o) for o in geo.get("openings", [])],
                               [tuple(s) for s in geo.get("shelving", [])])
        except (ValueError, KeyError, TypeError) as exc:
            self.note("room", f"room geometry unusable: {exc}")
            return Room()
        for w in fig.warnings:
            self.note("room", w)
        return Room(**fig.as_room_fields(), scale_method=geo.get("scale_method", ""), confidence=float(geo.get("confidence", 0)))

    def _save_stage_outputs(self) -> None:
        def save(name, data):
            (self.out / name).write_text(json.dumps(data, indent=2, default=str))
        for name, data in (("rows.json", self.journal["rows"]), ("room_frames.json", self.journal["room_frames"]),
                           ("assertions.json", self.journal["assertions"]),
                           ("remarks.json", [{k: v for k, v in r.items() if k != "at"} for r in self.remarks])):
            (self.root / name).write_text(json.dumps(data, indent=2))
        save("rows.json", sorted(self.row_info, key=lambda r: r["shelf"]))
        save("reads.json", {j.book.id: asdict(j.text) if j.text else None for j in self.books})
        save("resolved.json", {j.book.id: asdict(j.resolution) for j in self.books if j.resolution})
        save("timings.json", {"seconds": self.stats.seconds, "slowest_job": self.stats.slowest,
                              "model_calls": self.stats.calls,
                              "cost": self.deps.meter.cost() if self.deps.meter else None})


def run_pipeline(root: Path, deps: Deps, settings: Settings, on_event=None) -> ClaimPacket:
    """Replay a saved sweep through SweepRun: every row and room frame is submitted at once."""
    root = Path(root)
    meta = load(root / "sweep.json", {})
    locale = Locale.parse(meta.get("country") or settings.locale.country, meta.get("currency") or settings.locale.currency)
    run = SweepRun(root, deps, settings, locale, on_event)
    for a in load(root / "assertions.json", []):
        run.add_assertion(Assertion(**a))
    for r in load(root / "remarks.json", []):
        run.add_remark(r["kind"], r.get("said", ""), r.get("shelf", ""), r.get("title", ""), r.get("description", ""))
    for row in load(root / "rows.json", []):
        run.add_row(row["shelf"], row.get("frames", []))
    run.add_room_frames(load(root / "room_frames.json", []))
    scan = load(root / "scan.json", [])
    if scan:
        frames = []
        for f in scan:
            try:
                frames.append(ScanFrame(read_image(root, f["ref"]), f["alpha"], f["beta"], f["gamma"], f["ref"]))
            except FileNotFoundError:
                run.note("room", f"scan frame {f['ref']} missing")
        run.add_room_scan(frames)
    return run.finish(load(root / "room.json", None), meta)
