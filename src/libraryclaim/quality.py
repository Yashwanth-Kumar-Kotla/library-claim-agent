"""Per-frame capture checks. These run on every evidence frame during the sweep and decide
what the agent says out loud ("slow down", "there's glare"), so they must be cheap and
deterministic. Frames are scored at a fixed width so thresholds don't depend on the phone."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

WORK_WIDTH = 640


@dataclass(frozen=True)
class Thresholds:
    min_sharpness: float = 60.0   # variance of Laplacian at WORK_WIDTH
    max_glare: float = 0.04       # fraction of near-white pixels
    min_brightness: float = 45.0  # mean gray level
    # Global shift between consecutive frames (~1/3 s apart), px at WORK_WIDTH. Set from a real
    # handheld phone sweep: normal panning was ~18 px (median), 12 px flagged 64% of frames.
    max_motion: float = 45.0


@dataclass
class FrameQuality:
    sharpness: float
    glare: float
    brightness: float
    motion: float | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def to_work_gray(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("empty frame")
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w != WORK_WIDTH:
        gray = cv2.resize(gray, (WORK_WIDTH, max(1, round(h * WORK_WIDTH / w))), interpolation=cv2.INTER_AREA)
    return gray


def motion_between(prev: np.ndarray, cur: np.ndarray) -> float:
    """Global shift between frames, in px at WORK_WIDTH. A pan is mostly a translation, so
    phase correlation is enough and is far cheaper than dense flow. If the frames barely
    correlate the camera moved so far they no longer overlap: report that as infinite."""
    if prev.shape != cur.shape:
        cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]))
    window = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_64F)
    (dx, dy), response = cv2.phaseCorrelate(prev.astype(np.float64), cur.astype(np.float64), window)
    if response < 0.03:
        return float("inf")
    return float(np.hypot(dx, dy))


def assess(img: np.ndarray, prev_gray: np.ndarray | None = None, t: Thresholds = Thresholds()) -> tuple[FrameQuality, np.ndarray]:
    """Score one frame. Returns the quality and the work-size gray image to pass as prev_gray next time."""
    gray = to_work_gray(img)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float(np.count_nonzero(gray >= 250)) / gray.size
    brightness = float(gray.mean())
    motion = motion_between(prev_gray, gray) if prev_gray is not None else None

    problems = []
    if brightness < t.min_brightness:
        problems.append("too_dark")
    elif sharpness < t.min_sharpness:
        # A dark frame is also low-contrast; report the cause, not the symptom.
        problems.append("blurry")
    if glare > t.max_glare:
        problems.append("glare")
    if motion is not None and motion > t.max_motion:
        problems.append("moving_too_fast")
    return FrameQuality(sharpness, glare, brightness, motion, problems), gray


SPOKEN = {
    "too_dark": "it's too dark here, can you add some light",
    "blurry": "that was blurry, hold steady for a second",
    "glare": "there's glare on the spines, tilt the phone a little",
    "moving_too_fast": "slow down a little, I can't read the spines at that speed",
}


def coaching(problems: list[str]) -> str:
    """Short phrase for the live agent. Most important problem first."""
    order = ["moving_too_fast", "too_dark", "blurry", "glare"]
    for p in order:
        if p in problems:
            return SPOKEN[p]
    return ""
