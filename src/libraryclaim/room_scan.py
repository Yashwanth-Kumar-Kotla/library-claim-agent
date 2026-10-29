"""Room size from a turn-around scan: the policyholder stands near the middle and turns once.

Each frame comes with the phone's orientation (the browser's deviceorientation angles) and gets a
metric depth map from a pretrained model. Back-projecting the pixels and rotating them by the
phone's orientation puts every frame into one room-fixed frame. Then:

  - floor and ceiling are the lowest and highest dense horizontal layers of points: height;
  - looking out along each compass direction, the farthest surface is the wall (furniture stands
    in front of walls), which traces the floor outline: area, length, width, shape;
  - a printed marker of known size seen in any depth-mapped frame corrects the depth model's scale.

No phone browser exposes LiDAR, so this is how the sweep itself yields metric room figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .scale import find_markers

Depth = Callable[[np.ndarray], tuple[np.ndarray, float]]  # BGR image -> (depth in metres, focal length in px)

BINS = 16                 # frames kept per turn, one per 22.5 degrees of heading
MIN_COVERAGE_DEG = 300    # less than this and the outline has a hole we would have to invent
ARM_M = 0.25              # phone is held this far in front of the body's turning axis
AZ_STEP_DEG = 2
STRIDE = 6                # sample every Nth pixel when back-projecting


@dataclass
class ScanFrame:
    image: np.ndarray
    alpha: float   # deviceorientation degrees: compass-like heading
    beta: float    # front-back tilt (90 = phone upright)
    gamma: float   # left-right tilt
    ref: str = ""


@dataclass
class ScanResult:
    corners: list[list[float]]
    height_m: float | None
    scale_factor: float
    confidence: float
    coverage_deg: float
    notes: list[str] = field(default_factory=list)
    method: str = ""

    def as_room_json(self, shelving: list[tuple[float, float]] = ()) -> dict:
        return {"corners": self.corners, "height_m": self.height_m, "shelving": [list(s) for s in shelving],
                "scale_method": self.method, "confidence": self.confidence, "notes": self.notes}


def rotation(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Device frame -> room frame, per the W3C DeviceOrientation spec (Z-X'-Y'' intrinsic).
    Device frame: x to the right of the screen, y to its top, z out of the screen."""
    a, b, g = np.radians([alpha, beta, gamma])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    ry = np.array([[np.cos(g), 0, np.sin(g)], [0, 1, 0], [-np.sin(g), 0, np.cos(g)]])
    return rz @ rx @ ry


def back_project(depth: np.ndarray, focal: float, R: np.ndarray, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of a portrait back-camera frame -> 3D points in the room frame (camera at the origin)
    and how vertical each point's surface normal is (1 = floor or ceiling, 0 = wall).
    The back camera looks along -z of the device; image right is device +x, image down is device -y."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[v, u].astype(np.float64)
    grid = np.stack([(u - w / 2) * z / focal, -(v - h / 2) * z / focal, -z], axis=-1) @ R.T
    n = np.cross(grid[1:, 1:] - grid[1:, :-1], grid[1:, 1:] - grid[:-1, 1:])
    vertical = np.abs(n[..., 2]) / np.maximum(np.linalg.norm(n, axis=-1), 1e-9)
    pts, z = grid[1:, 1:], z[1:, 1:]
    ok = np.isfinite(z) & (z > 0.2) & (z < 15) & np.isfinite(vertical)
    return pts[ok], vertical[ok]


def horizontal_layer(z: np.ndarray, lowest: bool) -> float | None:
    """Height of the lowest (floor) or highest (ceiling) substantial horizontal surface. Desk and
    bed tops are horizontal too, so take the extreme layer that has enough points, not the median."""
    if len(z) < 200:
        return None
    lo, hi = np.floor(z.min() * 20) / 20, np.ceil(z.max() * 20) / 20
    hist, edges = np.histogram(z, bins=max(1, int(round((hi - lo) * 20))), range=(lo, hi))
    need = max(50, 0.05 * len(z))
    order = range(len(hist)) if lowest else range(len(hist) - 1, -1, -1)
    for i in order:
        if hist[i] >= need:
            sel = z[(z >= edges[i] - 0.05) & (z <= edges[i + 1] + 0.05)]
            return float(np.median(sel))
    return None


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_frames(frames: list[ScanFrame], bins: int = BINS) -> tuple[list[ScanFrame], float]:
    """The sharpest roughly-level frame in each heading sector, plus the most tilted-up and
    tilted-down frames (an upright phone sees about +/-31 degrees, often neither floor nor ceiling
    from mid-room), and how much of the circle the turn covered."""
    level = [f for f in frames if abs(f.beta - 90) <= 20]
    best: dict[int, tuple[float, ScanFrame]] = {}
    for f in level:
        k = int((f.alpha % 360) // (360 / bins))
        sharp = sharpness(f.image)
        if k not in best or sharp > best[k][0]:
            best[k] = (sharp, f)
    covered = len(best) * 360 / bins
    tilted = [f for f in frames if 25 <= abs(f.beta - 90) <= 70]
    up = sorted((f for f in tilted if f.beta > 90), key=lambda f: -f.beta)[:2]
    down = sorted((f for f in tilted if f.beta < 90), key=lambda f: f.beta)[:2]
    return [best[k][1] for k in sorted(best)] + up + down, covered


def marker_scale(image: np.ndarray, depth: np.ndarray, focal: float, size_m: float) -> float | None:
    """Ratio true/measured marker side, measuring the marker in 3D from the depth map. The depth
    model's error is mostly one shared scale, which this removes."""
    markers = find_markers(image)
    if not markers:
        return None
    m = max(markers, key=lambda m: m.side_px)
    if m.side_px < 40:
        return None
    h, w = depth.shape
    pts = []
    for x, y in m.corners:
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        z = float(np.median(depth[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]))
        pts.append([(x - w / 2) * z / focal, -(y - h / 2) * z / focal, -z])
    pts = np.array(pts)
    side = np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)])
    if side <= 0:
        return None
    s = size_m / side
    return s if 0.6 <= s <= 1.6 else None


def outline(points: np.ndarray, floor_z: float, ceil_z: float | None) -> np.ndarray:
    """Floor outline as the farthest wall point in each direction, smoothed. Returns (N, 2) in metres."""
    top = (ceil_z - 0.3) if ceil_z is not None else floor_z + 2.0
    band = points[(points[:, 2] > floor_z + 0.3) & (points[:, 2] < top)]
    r = np.hypot(band[:, 0], band[:, 1])
    az = (np.degrees(np.arctan2(band[:, 1], band[:, 0])) + 360) % 360
    nbins = 360 // AZ_STEP_DEG
    idx = (az // AZ_STEP_DEG).astype(int)
    radius = np.full(nbins, np.nan)
    for k in range(nbins):
        rk = r[idx == k]
        if len(rk) >= 5:
            # The wall is the far cluster in this direction (furniture stands in front of it). Its
            # median, not a high percentile: percentiles drift outward with noisy depth.
            far = np.percentile(rk, 95)
            radius[k] = np.median(rk[(rk >= 0.85 * far) & (rk <= 1.05 * far)])
    good = np.isfinite(radius)
    if good.sum() < nbins * 0.5:
        raise ValueError("too few directions saw a wall")
    k = np.arange(nbins)
    radius = np.interp(k, k[good], radius[good], period=nbins)  # fill small gaps around the circle
    padded = np.concatenate([radius[-2:], radius, radius[:2]])
    radius = np.array([np.median(padded[i:i + 5]) for i in range(nbins)])
    theta = np.radians(k * AZ_STEP_DEG + AZ_STEP_DEG / 2)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def rectify(xy: np.ndarray) -> tuple[list[list[float]], str]:
    """Corners for the room: a rectangle when the outline is one, else a simplified polygon."""
    edges = np.diff(np.vstack([xy, xy[:1]]), axis=0)
    ang = np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2)
    weights = np.hypot(edges[:, 0], edges[:, 1])
    hist, bins = np.histogram(ang, bins=45, range=(0, np.pi / 2), weights=weights)
    theta = bins[int(np.argmax(hist))] + (bins[1] - bins[0]) / 2   # dominant wall direction
    c, s = np.cos(-theta), np.sin(-theta)
    rot = xy @ np.array([[c, -s], [s, c]]).T
    x0, x1 = np.percentile(rot[:, 0], [2, 98])
    y0, y1 = np.percentile(rot[:, 1], [2, 98])
    poly_area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
    if poly_area / ((x1 - x0) * (y1 - y0)) >= 0.85:
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        c, s = np.cos(theta), np.sin(theta)
        return (box @ np.array([[c, -s], [s, c]]).T).round(3).tolist(), "rectangle fitted to the scanned outline"
    simple = cv2.approxPolyDP(xy.astype(np.float32).reshape(-1, 1, 2), 0.15, True).reshape(-1, 2)
    return simple.round(3).tolist(), f"non-rectangular outline simplified to {len(simple)} corners"


def measure_room(frames: list[ScanFrame], depth: Depth, marker_size_m: float = 0.15,
                 calibration: list[tuple[np.ndarray, np.ndarray, float]] = ()) -> ScanResult:
    """frames: one turn of the room. calibration: extra (image, depth, focal) seen elsewhere in the
    sweep that may contain the marker (e.g. a shelf row)."""
    picked, covered = pick_frames(frames)
    if covered < MIN_COVERAGE_DEG:
        raise ValueError(f"the turn covered about {covered:.0f} degrees; at least {MIN_COVERAGE_DEG} needed")
    notes, scales, clouds, flats = [], [], [], []
    for f in picked:
        d, focal = depth(f.image)
        R = rotation(f.alpha, f.beta, f.gamma)
        pts, vertical = back_project(d, focal, R)
        # The camera sits ARM_M in front of the turning axis: measure from the axis, not the lens.
        fwd = R @ np.array([0.0, 0.0, -1.0])
        fwd_h = np.array([fwd[0], fwd[1], 0.0])
        if np.linalg.norm(fwd_h) > 1e-6:
            pts = pts + ARM_M * fwd_h / np.linalg.norm(fwd_h)
        clouds.append(pts)
        flats.append(vertical > 0.9)
        s = marker_scale(f.image, d, focal, marker_size_m)
        if s:
            scales.append(s)
    for img, d, focal in calibration:
        s = marker_scale(img, d, focal, marker_size_m)
        if s:
            scales.append(s)
    scale = float(np.median(scales)) if scales else 1.0
    if not scales:
        notes.append("no marker seen in depth-mapped frames; depth model scale used uncorrected")
    pts = np.vstack(clouds) * scale
    flat = np.concatenate(flats)

    z = pts[:, 2]
    floor_z = horizontal_layer(z[flat & (z < -0.3)], lowest=True)
    ceil_z = horizontal_layer(z[flat & (z > 0.3)], lowest=False)
    height = round(ceil_z - floor_z, 2) if floor_z is not None and ceil_z is not None else None
    if height is None:
        notes.append("floor or ceiling not seen well enough to measure height (tilt the phone down and up during the turn)")
    elif not 1.8 <= height <= 6.0:
        notes.append(f"measured height {height} m is implausible; not reported")
        height = None

    # Without a measured floor, keep wall points from a band around eye level only.
    xy = outline(pts, floor_z if floor_z is not None else float(np.percentile(z, 2)), ceil_z)
    corners, shape = rectify(xy)
    notes.append(shape)
    confidence = round(min(1.0, covered / 360) * (0.85 if scales else 0.6) * (1.0 if height else 0.7), 2)
    method = (f"turn-around scan: Depth Pro metric depth on {len(picked)} frames, phone orientation, "
              + (f"scale corrected x{scale:.3f} from the printed marker" if scales else "no marker correction"))
    return ScanResult(corners, height, round(scale, 4), confidence, covered, notes, method)


def item_size_cm(box: tuple[float, float, float, float], depth: np.ndarray, focal: float,
                 scale: float = 1.0) -> tuple[float, float] | None:
    """Approximate width and height of an object's front face, whole centimetres."""
    size = front_size_cm(box, depth, focal, scale)
    return (round(size[0], 0), round(size[1], 0)) if size else None


def front_size_cm(box: tuple[float, float, float, float], depth: np.ndarray, focal: float,
                  scale: float = 1.0) -> tuple[float, float] | None:
    """Width and height of an object's front face from its box and the depth inside it."""
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = depth.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    cx0, cx1 = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4   # central part: the object, not what is behind it
    cy0, cy1 = y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
    z = float(np.median(depth[cy0:cy1, cx0:cx1])) * scale
    if not np.isfinite(z) or z <= 0:
        return None
    wc, hc = (x1 - x0) * z / focal * 100, (y1 - y0) * z / focal * 100
    if wc > 600 or hc > 400:
        return None
    return wc, hc
