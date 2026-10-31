"""State of one live sweep: what the phone sends, what the agent is told, what the pipeline gets.

Every full-resolution frame is saved (so every frame_ref opens) and scored for blur, glare,
darkness and speed. While the agent has a shelf row open, frames belong to that row; when the
agent closes it, a few keyframes spanning the row go to the pipeline and processing starts at
once. Frames taken with no row open are room views: the sharpest frame of each short window is
sent for non-book item detection.

Nothing here talks to a model or the network directly, so it is all testable offline.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..config import Locale, Settings
from ..pipeline import Deps, SweepRun
from ..quality import WORK_WIDTH, FrameQuality, assess, coaching
from ..review import Assertion
from ..room_scan import BINS, MIN_COVERAGE_DEG, ScanFrame, aim, best_marker

ROW_NAME = re.compile(r"^[A-Z][1-9][0-9]?$")
KEYFRAME_STEP = 0.35        # keep a keyframe each time the camera has moved ~35% of a frame width
ROOM_WINDOW_S = 2.5         # at most one room frame per window of this length ...
ROOM_MIN_TRAVEL = 0.5       # ... and only once the view has moved half a frame width since the last one
# Capture feedback is mostly silent. A real phone sweep with the agent told about every bad
# stretch was unusable: 13 of its 23 utterances in 96 s were app-triggered, and it never waited.
SCREEN_WINDOW = 4           # banner: shown when 3 of the last 4 frames share a problem
SPEAK_WINDOW_S = 6.0        # voice: only when a problem has lasted this long ...
SPEAK_SHARE = 0.7           # ... on most frames in that time ...
SPEAK_MIN_FRAMES = 10
SPEAK_GAP_S = 20.0          # ... at least this long after the last thing the app had said ...
QUIET_START_S = 10.0        # ... never in the opening seconds, and once per shelf per problem.
SPINE_PROBLEMS = {"moving_too_fast", "blurry", "glare"}  # only matter while a shelf is being read
NUDGE_AFTER_S = 30.0        # camera moving this long with no shelf open: ask the agent about it
NUDGE_REPEAT_S = 45.0
# The turn is guided on screen (a ring that fills, a tilt bar), so the agent hears only the edges:
# when the turn is complete, and once per scan if they stop with gaps left. Rules match the phone page.
LEVEL_DEG, TILT_DEG = 20, 25  # roughly level for the ring; tilted this far up or down for floor/ceiling
SCAN_STALL_S = 10.0          # no new sector or tilt for this long, once they have started turning ...
SCAN_START_GRACE_S = 25.0    # ... or this long after the scan opened (the agent may still be explaining)
MIN_USABLE_PER_ROW = 2


@dataclass
class Frame:
    ref: str
    t: float
    quality: FrameQuality
    sharpness: float


@dataclass
class OpenRow:
    shelf: str
    started: float
    frames: list[Frame] = field(default_factory=list)


class LiveSweep:
    def __init__(self, root: Path, deps: Deps, settings: Settings, on_event=None, clock=time.monotonic):
        self.root = Path(root)
        (self.root / "frames").mkdir(parents=True, exist_ok=True)
        self.settings, self.deps = settings, deps
        self.clock = clock
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.t0 = clock()
        self.on_event = on_event or (lambda kind, data: None)
        self.locale = settings.locale
        self.run = SweepRun(self.root, deps, settings, self.locale, self.on_event)
        self.lock = threading.Lock()
        self.n = 0
        self.prev_gray: np.ndarray | None = None
        self.recent: deque[tuple[float, FrameQuality]] = deque()
        self.last_spoken = -1e9
        self.said_this_row: set[str] = set()
        self.row: OpenRow | None = None
        self.rows_done: list[str] = []
        self.room_window: list[Frame] = []
        self.room_window_start: float | None = None
        self.room_frames: list[str] = []
        self.room_travel = float("inf")  # the first room view is always taken
        self.ended = False
        self.result: dict | None = None   # end_sweep's answer, repeated to any later call
        self.row_closed_at = 0.0
        self.scanning = False
        self.scan_entries: list[dict] = []   # {"ref", "alpha", "beta", "gamma"} for frames taken during the turn
        self.orientation_seen = False
        self.last_nudge = -1e9
        self.scan_said: set[str] = set()     # "complete", "stalled": each said at most once per scan
        self.scan_marker = False             # the printed marker was in view during the turn: exact room scale
        self.scan_progress = (0, False, False)
        self.scan_progress_at = self.scan_started_at = 0.0

    # ---- locale ----

    def set_locale(self, country: str, currency: str) -> dict:
        try:
            locale = Locale.parse(country, currency)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if self.run.books or self.run.rows:
            return {"ok": False, "error": "locale can only be set before the first shelf"}
        self.locale = self.run.locale = locale
        return {"ok": True, "country": locale.country, "currency": locale.currency}

    # ---- frames ----

    def add_frame(self, jpeg: bytes, orientation: tuple[float, float, float] | None = None) -> tuple[str | None, str | None]:
        """Save and score one full-resolution frame. Returns (spoken, screen): a remark for the
        agent to say, which is rare, and a hint for the on-screen banner, which costs nothing."""
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        with self.lock:
            if self.ended:
                return None, None
            self.n += 1
            ref = f"frames/{self.n:05d}.jpg"
            prev = self.prev_gray
        (self.root / ref).write_bytes(jpeg)
        q, gray = assess(img, prev)
        # A marker seen during the turn turns the room's depth into exact metres; without one the
        # scale is the depth model's own. Looked for only until it is found once.
        if self.scanning and not self.scan_marker and orientation is not None:
            m = best_marker(img)
            if m and m.side_px >= 40:
                self.scan_marker = True
                self.on_event("scan_marker", {})
        now = self.clock()
        frame = Frame(ref, now - self.t0, q, q.sharpness)
        with self.lock:
            self.prev_gray = gray
            if orientation is not None:
                self.orientation_seen = True
                if self.scanning:
                    a, b, g = orientation
                    self.scan_entries.append({"ref": ref, "alpha": a, "beta": b, "gamma": g})
                    bins, up, down = scan_coverage(self.scan_entries)
                    if (len(bins), up, down) != self.scan_progress:
                        self.scan_progress, self.scan_progress_at = (len(bins), up, down), now
            self.recent.append((now, q))
            while self.recent and now - self.recent[0][0] > SPEAK_WINDOW_S:
                self.recent.popleft()
            if self.row is not None:
                self.row.frames.append(frame)
            else:
                self._room_frame(frame)
            return self._coaching(now)

    def nudge(self) -> str | None:
        """A reminder for the agent when the camera has been moving with no shelf open: in a
        real sweep the agent once left shelves unopened for a minute, so the books were never
        counted. Only after the first shelf and before the end, and not too often."""
        with self.lock:
            if self.row is not None or self.scanning or not self.rows_done or self.ended:
                return None
            now = self.clock()
            if now - self.row_closed_at < NUDGE_AFTER_S or now - self.last_nudge < NUDGE_REPEAT_S:
                return None
            if now - self.last_spoken < SPEAK_GAP_S:
                return None
            if self.room_travel < 0.5 and not self.room_frames:
                return None
            self.last_nudge = self.last_spoken = now
        return ("No shelf is open, but the camera is moving. If they are filming books, call start_row "
                "for the shelf in view now; if they are showing the rest of the room, carry on.")

    def scan_prompt(self) -> str | None:
        """What the agent should hear about the turn, if anything: it is complete (once), or the
        policyholder has stopped with parts missing (once). Directions are the screen's job."""
        with self.lock:
            if not self.scanning or not self.scan_entries:
                return None
            n, up, down = self.scan_progress
            now = self.clock()
            covered = n * 360 / BINS
            if covered >= MIN_COVERAGE_DEG and up and down:
                if "complete" in self.scan_said:
                    return None
                self.scan_said.add("complete")
                if not self.scan_marker:
                    return (f"The room turn is complete ({covered:.0f} degrees, tilted up and down), but the printed "
                            "marker was not in view. Ask once, briefly: if there is a marker on a wall, point the phone "
                            "at it for a second. As soon as they have, or if there is none, call finish_room_scan.")
                return f"The room turn is complete ({covered:.0f} degrees, tilted up and down). Call finish_room_scan now."
            started = n >= 2 or now - self.scan_started_at >= SCAN_START_GRACE_S
            if "stalled" in self.scan_said or not started or now - self.scan_progress_at < SCAN_STALL_S:
                return None
            self.scan_said.add("stalled")
            self.last_spoken = now
        missing = [f"about {360 - covered:.0f} degrees of the turn"] if covered < MIN_COVERAGE_DEG else []
        missing += [w for w, done in (("the tilt up to the ceiling", up), ("the tilt down to the floor", down)) if not done]
        return (f"The room turn has stalled; still missing {' and '.join(missing)}. Give one short, calm "
                "reminder to follow the ring on the screen. Do not give directions.")

    def _relevant(self, q: FrameQuality) -> list[str]:
        # Speed, blur and glare only matter while a shelf is being read; darkness always does.
        return [p for p in q.problems if self.row is not None or p not in SPINE_PROBLEMS]

    def _coaching(self, now: float) -> tuple[str | None, str | None]:
        last = [self._relevant(q) for _, q in list(self.recent)[-SCREEN_WINDOW:]]
        screen_due = [p for p in {p for ps in last for p in ps} if sum(p in ps for ps in last) >= 3]
        screen = coaching(screen_due) or None

        spoken = None
        window = [self._relevant(q) for _, q in self.recent]
        if (len(window) >= SPEAK_MIN_FRAMES and now - self.t0 >= QUIET_START_S
                and now - self.last_spoken >= SPEAK_GAP_S):
            lasting = [p for p in {p for ps in window for p in ps}
                       if sum(p in ps for ps in window) / len(window) >= SPEAK_SHARE and p not in self.said_this_row]
            if lasting:
                spoken = coaching(lasting)
                self.last_spoken = now
                self.said_this_row.update(lasting)
        return spoken, screen

    def _room_frame(self, frame: Frame) -> None:
        self.room_travel += (frame.quality.motion or 0.0) / WORK_WIDTH
        if self.room_window_start is None:
            self.room_window_start = frame.t
        self.room_window.append(frame)
        if frame.t - self.room_window_start >= ROOM_WINDOW_S:
            self._flush_room_window()

    def _flush_room_window(self) -> None:
        """Send the sharpest frame of the window for item detection, but only if the camera has
        moved on: a phone held still (or a laptop webcam) otherwise sends the same view again and
        again, and every repeat is a chance to count the same chair twice (seen: 100 'items' at one desk)."""
        usable = [f for f in self.room_window if "too_dark" not in f.quality.problems]
        if usable and self.room_travel >= ROOM_MIN_TRAVEL:
            self.room_travel = 0.0
            best = max(usable, key=lambda f: f.sharpness)
            self.room_frames.append(best.ref)
            self.run.add_room_frame(best.ref)
        self.room_window, self.room_window_start = [], None

    # ---- rows, driven by the agent ----

    def start_row(self, shelf: str) -> dict:
        shelf = shelf.strip().upper()
        if not ROW_NAME.match(shelf):
            return {"ok": False, "error": "row names are a unit letter and a shelf number from the top, e.g. A1"}
        with self.lock:
            closed = self._close_row() if self.row else None
            # Seen live: the turn had started when the policyholder said there was one more bookcase.
            # The shelf comes first; the turn is stopped and simply started again afterwards.
            stopped_scan = self.scanning
            if stopped_scan:
                self.scanning, self.scan_entries = False, []
            self._flush_room_window()
            self.row = OpenRow(shelf, self.clock() - self.t0)
            self.said_this_row = set()
        if stopped_scan:
            self.on_event("scan_rejected", {"advice": "the room turn was stopped to film a shelf"})
        out = {"ok": True, "row": shelf, "recapture": shelf in self.rows_done}
        if closed:
            out["previous_row"] = closed
        if stopped_scan:
            out["room_turn"] = "stopped for this shelf; call start_room_scan again once the shelves are done"
        self.on_event("row_open", {"shelf": shelf})
        return out

    def finish_row(self) -> dict:
        with self.lock:
            if self.row is None:
                return {"ok": False, "error": "no row is open; call start_row first"}
            return self._close_row()

    def _close_row(self) -> dict:
        row, self.row = self.row, None
        self.row_closed_at = self.clock()
        usable = [f for f in row.frames if not ({"blurry", "too_dark", "moving_too_fast"} & set(f.quality.problems))]
        report = {"row": row.shelf, "frames": len(row.frames), "usable_frames": len(usable)}
        if len(usable) < MIN_USABLE_PER_ROW:
            bad = sorted({p for f in row.frames for p in f.quality.problems})
            report.update(ok=False, advice=f"too few clear frames of {row.shelf} ({', '.join(bad) or 'too short'}); "
                                          f"ask the policyholder to go back over {row.shelf} slowly")
            self.on_event("row_rejected", report)
            return report
        keys = keyframes(usable)
        self.run.add_row(row.shelf, [f.ref for f in keys])
        if row.shelf not in self.rows_done:
            self.rows_done.append(row.shelf)
        report.update(ok=True, keyframes=len(keys), rows_done=list(self.rows_done))
        glare = sum("glare" in f.quality.problems for f in usable) / len(usable)
        if glare > 0.5:
            report["advice"] = "glare on most frames of this row; some spines may be unreadable"
        self.on_event("row_closed", report)
        return report

    # ---- the turn-around scan for room size ----

    def start_room_scan(self) -> dict:
        with self.lock:
            closed = self._close_row() if self.row else None
            self.scanning, self.scan_entries, self.scan_marker = True, [], False
            self.scan_said, self.scan_progress = set(), (0, False, False)
            self.scan_started_at = self.scan_progress_at = self.clock()
        self.on_event("scan_open", {})
        out = {"ok": True, "instructions": "stand near the middle of the room, hold the phone upright at chest height, "
                                          "and follow the ring on the screen: turn slowly all the way round, letting it "
                                          "pass over the marker on the wall if there is one, then tilt up to where the "
                                          "wall meets the ceiling and down to where it meets the floor. The app tells "
                                          "you when the turn is complete."}
        if closed:
            out["previous_row"] = closed
        return out

    def finish_room_scan(self) -> dict:
        """Check the turn now, while the policyholder is still standing there. A turn that is only missing
        part of the circle or a tilt stays open, so they just carry on (seen live: a full 360-degree turn
        thrown away for want of a tilt, and the policyholder gave up rather than redo it)."""
        with self.lock:
            if not self.scanning:
                return {"ok": False, "error": "no room scan is open; call start_room_scan first"}
            entries = list(self.scan_entries)
            if entries:
                bins, up, down = scan_coverage(entries)
                covered = len(bins) * 360 / BINS
                missing = []
                if covered < MIN_COVERAGE_DEG:
                    missing.append(f"turn the rest of the way round (about {360 - covered:.0f} degrees left)")
                if not up:
                    missing.append("tilt the phone up to the ceiling")
                if not down:
                    missing.append("tilt it down to the floor")
                if missing:
                    return {"ok": False, "still_open": True, "covered_deg": covered,
                            "advice": f"The turn is not finished and is still open: ask them to {' and '.join(missing)}, "
                                      "following the screen. The app tells you when it is complete; do not call "
                                      "finish_room_scan again before that."}
            self.scanning = False
            self.row_closed_at = self.clock()
        if not entries:
            msg = ("this phone did not share its orientation, so room size cannot be measured; carry on"
                   if not self.orientation_seen else "no frames were captured during the turn; call start_room_scan and ask them to do it again")
            self.on_event("scan_rejected", {"advice": msg})
            return {"ok": False, "advice": msg}
        n = self._measure(entries)
        self.on_event("scan_closed", {"covered_deg": covered})
        return {"ok": True, "covered_deg": covered, "frames": n, "note": "room size is being measured"}

    def _measure(self, entries: list[dict]) -> int:
        (self.root / "scan.json").write_text(json.dumps(entries, indent=1))
        frames = []
        for e in entries:
            img = cv2.imread(str(self.root / e["ref"]))
            if img is not None:
                frames.append(ScanFrame(img, e["alpha"], e["beta"], e["gamma"], e["ref"]))
        self.run.add_room_scan(frames)
        return len(frames)

    # ---- what the policyholder says ----

    def record_remark(self, kind: str, said: str, title: str = "", description: str = "", shelf: str = "") -> dict:
        shelf = (shelf or (self.row.shelf if self.row else (self.rows_done[-1] if self.rows_done else ""))).upper()
        if kind in ("skip_shelf",):
            if not shelf:
                return {"ok": False, "error": "which shelf? name it like A2"}
            self.run.add_assertion(Assertion("skip_shelf", shelf=shelf, said=said))
            return {"ok": True, "applied_to": shelf}
        if kind in ("first_edition", "signed", "rare"):
            if not title:
                return {"ok": False, "error": "ask which book (its title) before recording this"}
            self.run.add_remark(kind, said, shelf=shelf, title=title)
            return {"ok": True, "will_match": f"'{title}' on {shelf or 'any shelf'}"}
        if kind in ("is_print", "original", "not_mine"):
            if kind == "not_mine":
                self.run.add_remark("not_mine", said, description=description, title=title, shelf=shelf)
                return {"ok": True}
            self.run.add_remark(kind, said, description=description)
            return {"ok": True, "will_match": description or "the artwork"}
        return {"ok": False, "error": f"unknown remark kind {kind}"}

    # ---- coverage and the end ----

    def gaps(self) -> list[str]:
        """Rows skipped inside a unit: A1, A2, A4 done means A3 is missing."""
        by_unit: dict[str, list[int]] = {}
        for r in self.rows_done:
            by_unit.setdefault(r[0], []).append(int(r[1:]))
        missing = []
        for unit, nums in sorted(by_unit.items()):
            missing += [f"{unit}{n}" for n in range(1, max(nums) + 1) if n not in nums]
        return missing

    def end(self, confirmed: bool = False, closed_early: bool = False) -> dict:
        with self.lock:
            if self.ended:
                # The agent sometimes calls end_sweep twice (on its own and again when End is tapped).
                # A plain error made it tell a policyholder the sweep "has already ended" mid-build.
                return self.result or {"ok": True, "status": "finishing",
                                       "note": "the claim packet is already being built; its summary comes back from "
                                               "the first end_sweep call. Say nothing about the result until then."}
            if self.row:
                self._close_row()
            scan_open = self.scanning
        if scan_open:
            self.finish_room_scan()
        with self.lock:
            unfinished = list(self.scan_entries) if self.scanning else None
            self.scanning = False
        if unfinished is not None:
            # Ended mid-turn: a full circle without the tilt still gives the floor and walls (height unmeasured).
            if len(scan_coverage(unfinished)[0]) * 360 / BINS >= MIN_COVERAGE_DEG:
                self._measure(unfinished)
            self.on_event("scan_closed", {})
        with self.lock:
            missing = self.gaps()
            if missing and not confirmed:
                return {"ok": False, "missing_rows": missing,
                        "advice": f"rows {', '.join(missing)} were never captured; ask before ending"}
            self._flush_room_window()
            self.ended = True
        self.on_event("finishing", {})
        if closed_early:
            self.run.note("sweep", "connection closed before the sweep was ended; packet built from what was captured")
        meta = {"id": self.root.name, "captured_at": self.started_at, "device": "phone browser",
                "duration_s": round(self.clock() - self.t0, 1), "capture_path": "browser"}
        (self.root / "sweep.json").write_text(json.dumps(meta | {"country": self.locale.country,
                                                                  "currency": self.locale.currency}, indent=2))
        packet = self.run.finish(None, meta)
        # Keep what the policyholder saw: replays rewrite claim_packet.json, this copy stays.
        (self.root / "out" / "claim_packet.live.json").write_text((self.root / "claim_packet.json").read_text())
        t = packet.totals
        cur = self.locale.currency
        summary = (f"{t.book_count} books counted, {t.books_identified} identified, {t.books_unidentified} not readable "
                   f"or not certain. Books: replacement {cur} {t.books_replacement_cost:,.2f}, used value {cur} "
                   f"{t.books_used_value:,.2f}. Other contents: {len(packet.items)} items, {cur} "
                   f"{t.items_replacement_cost_low:,.2f} to {t.items_replacement_cost_high:,.2f}. "
                   f"{t.excluded_from_totals} lines excluded from totals and {len(packet.review_queue)} notes for "
                   f"the adjuster to review.")
        if packet.room.floor_area_m2 is None:
            summary += " Room size was not measured in this sweep."
        figures = {"books": t.book_count, "identified": t.books_identified, "items": len(packet.items),
                   "floor_m2": packet.room.floor_area_m2, "currency": cur}  # for the phone's finishing screen
        self.result = {"ok": True, "summary": summary, "figures": figures,
                       "report": f"/sweeps/{self.root.name}/report.html"}
        return self.result


def scan_coverage(entries: list[dict]) -> tuple[set[int], bool, bool]:
    """Heading sectors seen with the camera roughly level, and whether it was tilted up and down."""
    aims = [aim(e["alpha"], e["beta"], e.get("gamma") or 0.0) for e in entries]
    bins = {int(h // (360 / BINS)) for h, p in aims if abs(p) <= LEVEL_DEG}
    up = any(p >= TILT_DEG for _, p in aims)
    down = any(-p >= TILT_DEG for _, p in aims)
    return bins, up, down


def keyframes(frames: list[Frame], step: float = KEYFRAME_STEP) -> list[Frame]:
    """Frames spaced along the pan: a new one each time the camera has moved about `step` of a
    frame width, picking the sharpest frame in each stretch, plus the last frame. Enough overlap
    to stitch, few enough to stitch quickly."""
    if len(frames) <= 2:
        return list(frames)
    out, travelled, best = [frames[0]], 0.0, None
    for f in frames[1:]:
        travelled += (f.quality.motion or 0.0) / WORK_WIDTH
        if best is None or f.sharpness > best.sharpness:
            best = f
        if travelled >= step:
            out.append(best)
            travelled, best = 0.0, None
    if out[-1] is not frames[-1]:
        out.append(frames[-1] if best is None else best)
    return out
