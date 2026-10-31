"""Room size from a turn-around scan: the policyholder stands near the middle and turns once.

Each frame comes with the phone's orientation (the browser's deviceorientation angles) and gets a
metric depth map from a pretrained model. Back-projecting the pixels and rotating them by the
phone's orientation puts every frame into one room-fixed frame. Then:

  - floor and ceiling are the lowest and highest dense horizontal layers of points: height;
  - looking out along each compass direction, the farthest surface is the wall (furniture stands
    in front of walls), which traces the floor outline: area, length, width, shape;
  - metric scale: a printed marker of known size, if any frame of the turn (or a shelf) saw one;
    otherwise the phone's height above the floor. Layout methods such as HorizonNet fix the camera
    height for the same reason. Unlike a panorama, every phone frame gets its own depth map with its
    own scale error, so the floor is measured in the frames tilted down at it and the ceiling in each
    frame tilted up, and the median is kept rather than one extreme;
  - the outline is cleaned under the Manhattan assumption (walls meet at right angles): dents and
    spikes smaller than furniture are removed, a real L-shape is kept.

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
CAMERA_HEIGHT_M = 1.4     # phone held in front, chest to eye height, while turning
GRID_M = 0.05             # floor outline raster for the Manhattan cleanup
SPIKE_M = 0.5             # outward spikes narrower than this are depth noise or a glimpse through a door
DENT_M = 0.5              # inward dents narrower than this are furniture in front of a wall


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


def aim(alpha: float, beta: float, gamma: float) -> tuple[float, float]:
    """Heading and pitch (degrees) of the back camera's axis. With the phone upright the browser's
    angles are in gimbal lock: a sideways roll shows up in gamma as much as in alpha (one home turn
    had gamma from -55 to +84 degrees on level frames), so alpha alone is not where the camera points.
    Heading equals alpha when gamma is 0; pitch is 0 when level, positive tilted up."""
    fwd = rotation(alpha, beta, gamma) @ np.array([0.0, 0.0, -1.0])
    heading = round(float(np.degrees(np.arctan2(fwd[1], fwd[0])) - 90), 6) % 360   # 119.9999 is 120
    return heading, round(float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))), 6)


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
    """The sharpest roughly-level frame in each heading sector, plus tilted-up and tilted-down
    frames (an upright phone sees about +/-31 degrees, often neither floor nor ceiling from
    mid-room), and how much of the circle the turn covered."""
    aims = {id(f): aim(f.alpha, f.beta, f.gamma) for f in frames}
    level = [f for f in frames if abs(aims[id(f)][1]) <= 20]
    best: dict[int, tuple[float, ScanFrame]] = {}
    for f in level:
        k = int(aims[id(f)][0] // (360 / bins))
        sharp = sharpness(f.image)
        if k not in best or sharp > best[k][0]:
            best[k] = (sharp, f)
    covered = len(best) * 360 / bins
    # Floor and ceiling are measured per frame and the median kept, so take one tilted frame in each
    # third of the circle: one frame with a bad depth map is then outvoted. Prefer about 40 degrees:
    # steep enough to see the floor or ceiling, shallow enough to still see the walls it is aligned by
    # (a frame aimed at the feet shows no wall).
    tilted = [f for f in frames if 25 <= abs(aims[id(f)][1]) <= 75]
    extra = []
    for up in (True, False):
        side: dict[int, ScanFrame] = {}
        for f in tilted:
            if (aims[id(f)][1] > 0) == up:
                k = int(aims[id(f)][0] // 120)
                if k not in side or abs(abs(aims[id(f)][1]) - 40) < abs(abs(aims[id(side[k])][1]) - 40):
                    side[k] = f
        extra += [side[k] for k in sorted(side)]
    return [best[k][1] for k in sorted(best)] + extra, covered


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


def wall_radius(points: np.ndarray) -> np.ndarray:
    """Horizontal distance to the wall in each 2-degree direction (NaN where nothing was seen)."""
    r = np.hypot(points[:, 0], points[:, 1])
    az = (np.degrees(np.arctan2(points[:, 1], points[:, 0])) + 360) % 360
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
    return radius


def align_to_walls(wall_pts: np.ndarray, arm: np.ndarray, ref: np.ndarray) -> float | None:
    """Scale for one frame's depth that makes its walls agree with the walls the level frames saw.
    A wall is vertical, so its horizontal distance is the same at floor or ceiling height; a frame
    tilted at the ceiling still shows the tops of the walls. wall_pts are relative to the lens and
    arm is the lens's offset from the turning axis, which is a real length and is not rescaled."""
    k = 1.0
    for _ in range(4):
        own = wall_radius(wall_pts * k + arm)
        both = np.isfinite(own) & np.isfinite(ref) & (own > 0.3)
        if both.sum() < 5:
            return None
        k *= float(np.median(ref[both] / own[both]))
    return k if 0.5 <= k <= 2.0 else None


def outline(points: np.ndarray, floor_z: float, ceil_z: float | None) -> np.ndarray:
    """Floor outline as the farthest wall point in each direction, smoothed. Returns (N, 2) in metres."""
    top = (ceil_z - 0.3) if ceil_z is not None else floor_z + 2.0
    radius = wall_radius(points[(points[:, 2] > floor_z + 0.3) & (points[:, 2] < top)])
    nbins = len(radius)
    good = np.isfinite(radius)
    if good.sum() < nbins * 0.5:
        raise ValueError("too few directions saw a wall")
    k = np.arange(nbins)
    radius = np.interp(k, k[good], radius[good], period=nbins)  # fill small gaps around the circle
    padded = np.concatenate([radius[-2:], radius, radius[:2]])
    radius = np.array([np.median(padded[i:i + 5]) for i in range(nbins)])
    theta = np.radians(k * AZ_STEP_DEG + AZ_STEP_DEG / 2)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def dominant_direction(xy: np.ndarray) -> float:
    """Angle of the walls (mod 90 degrees), weighted by outline edge length."""
    edges = np.diff(np.vstack([xy, xy[:1]]), axis=0)
    ang = np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2)
    hist, bins = np.histogram(ang, bins=45, range=(0, np.pi / 2), weights=np.hypot(edges[:, 0], edges[:, 1]))
    return float(bins[int(np.argmax(hist))] + (bins[1] - bins[0]) / 2)


def rectify(xy: np.ndarray) -> tuple[list[list[float]], str]:
    """Corners for the room. The outline is turned to the wall directions and drawn on a 5 cm grid;
    an opening then closing removes dents and spikes narrower than DENT_M (a chair, a noisy depth
    patch) but keeps a notch the size of a room (an L-shape). A rectangle when the cleaned shape is
    one, else its simplified polygon."""
    theta = dominant_direction(xy)
    c, s = np.cos(-theta), np.sin(-theta)
    rot = xy @ np.array([[c, -s], [s, c]]).T
    lo = rot.min(axis=0) - 1.0
    px = np.round((rot - lo) / GRID_M).astype(np.int32)
    mask = np.zeros((px[:, 1].max() + 41, px[:, 0].max() + 41), np.uint8)
    cv2.fillPoly(mask, [px.reshape(-1, 1, 2)], 255)
    spike, dent = (np.ones((int(m / GRID_M),) * 2, np.uint8) for m in (SPIKE_M, DENT_M))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, spike), cv2.MORPH_CLOSE, dent)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("room outline vanished in cleanup")
    blob = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(blob)
    c, s = np.cos(theta), np.sin(theta)
    back = np.array([[c, -s], [s, c]]).T
    if not big_notch(blob, rect, mask.shape):
        box = cv2.boxPoints(rect) * GRID_M + lo
        return (box @ back).round(3).tolist(), "rectangle fitted to the scanned outline"
    simple = cv2.approxPolyDP(blob, 0.2 / GRID_M, True).reshape(-1, 2) * GRID_M + lo
    return (simple @ back).round(3).tolist(), f"non-rectangular outline simplified to {len(simple)} corners"


def big_notch(blob: np.ndarray, rect, shape: tuple[int, int]) -> bool:
    """Whether the outline is missing one large block of its bounding rectangle, as an L-shaped room
    is (one block, ~18% of the rectangle), rather than slivers along noisy walls (two home turns of a
    plain bedroom: largest gap ~10%, the rest spread over other walls)."""
    box = np.zeros(shape, np.uint8)
    cv2.fillPoly(box, [cv2.boxPoints(rect).astype(np.int32)], 255)
    inside = np.zeros(shape, np.uint8)
    cv2.fillPoly(inside, [blob], 255)
    k = np.ones((int(SPIKE_M / GRID_M),) * 2, np.uint8)
    missing = cv2.morphologyEx(cv2.subtract(box, inside), cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(missing)
    blocks = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    if not blocks:
        return False
    return blocks[0] >= 0.12 * rect[1][0] * rect[1][1] and blocks[0] >= 0.75 * sum(blocks)


def best_marker(image: np.ndarray):
    markers = find_markers(image)
    return max(markers, key=lambda m: m.side_px) if markers else None


def measure_room(frames: list[ScanFrame], depth: Depth, marker_size_m: float = 0.15,
                 calibration: list[tuple[np.ndarray, np.ndarray, float]] = ()) -> ScanResult:
    """frames: one turn of the room. calibration: extra (image, depth, focal) seen elsewhere in the
    sweep that may contain the marker (e.g. a shelf row)."""
    picked, covered = pick_frames(frames)
    if covered < MIN_COVERAGE_DEG:
        raise ValueError(f"the turn covered about {covered:.0f} degrees; at least {MIN_COVERAGE_DEG} needed")
    # The marker may be in a frame that was not picked for its heading: look at every frame of the turn.
    seen = sorted(((m.side_px, n) for n, f in enumerate(frames) for m in [best_marker(f.image)]
                   if m and m.side_px >= 40), reverse=True)
    picked += [frames[n] for _, n in seen[:2] if not any(frames[n] is p for p in picked)]

    notes, scales, clouds, arms, flats, walls, kinds = [], [], [], [], [], [], []
    for f in picked:
        d, focal = depth(f.image)
        R = rotation(f.alpha, f.beta, f.gamma)
        pts, vertical = back_project(d, focal, R)
        # The camera sits ARM_M in front of the turning axis: measure from the axis, not the lens.
        fwd = R @ np.array([0.0, 0.0, -1.0])
        fwd_h = np.array([fwd[0], fwd[1], 0.0])
        arms.append(ARM_M * fwd_h / np.linalg.norm(fwd_h) if np.linalg.norm(fwd_h) > 1e-6 else np.zeros(3))
        clouds.append(pts)
        flats.append(vertical > 0.9)
        walls.append(vertical < 0.3)
        pitch = aim(f.alpha, f.beta, f.gamma)[1]
        kinds.append("down" if pitch <= -20 else "up" if pitch >= 20 else "level")
        s = marker_scale(f.image, d, focal, marker_size_m)
        if s:
            scales.append(s)
    for img, d, focal in calibration:
        s = marker_scale(img, d, focal, marker_size_m)
        if s:
            scales.append(s)

    # Each depth map has its own scale error (one home turn: the frame tilted at the ceiling read
    # about twice as deep as the one tilted at the floor). Bring every tilted frame to the scale of
    # the level frames by matching walls, then measure floor and ceiling frame by frame.
    level = [i for i, k in enumerate(kinds) if k == "level"]
    aligned = set(level)
    if level:
        ref = wall_radius(np.vstack([clouds[i][walls[i]] + arms[i] for i in level]))
        for i, k in enumerate(kinds):
            if k != "level":
                a = align_to_walls(clouds[i][walls[i]], arms[i], ref)
                if a:
                    clouds[i] = clouds[i] * a
                    aligned.add(i)
    def layer(i: int, lowest: bool) -> float | None:
        z = clouds[i][:, 2]
        sel = flats[i] & ((z < -0.3) if lowest else (z > 0.3))
        return horizontal_layer(z[sel], lowest)
    # Frames not on the walls' scale could each be off by their own factor: used only if nothing else.
    floors = [(i, -fz) for i in range(len(picked)) if kinds[i] == "down" and i in aligned for fz in [layer(i, True)] if fz]
    if not floors:
        floors = [(i, -fz) for i in range(len(picked)) if kinds[i] == "down" for fz in [layer(i, True)] if fz]
        if floors:
            notes.append("the floor was only seen looking straight down, with no wall in view; scale less certain")
    prior = [CAMERA_HEIGHT_M / drop for _, drop in floors if 0.5 <= CAMERA_HEIGHT_M / drop <= 2.0]
    # A desk or bed top is horizontal too and can pass for the floor in one frame. Trust the phone's
    # height only when two or more frames agree on the floor (two home turns of one room: with one
    # frame, 13.8 and 34.5 m2; Depth Pro's own scale gave 17.3 and 20.0 m2).
    if len(prior) < 2 or max(prior) / min(prior) > 1.1:
        if prior:
            notes.append("the floor was not seen consistently in the frames tilted down; camera-height check skipped")
        prior = []
    if scales:
        scale = float(np.median(scales))
        how = f"scale corrected x{scale:.3f} from the printed marker"
        if prior:
            gap = float(np.median(prior)) / scale - 1
            notes.append(f"camera-height check {'agrees' if abs(gap) <= 0.1 else 'disagrees'} with the marker ({gap:+.0%})")
    elif prior:
        scale = float(np.median(prior))
        how = (f"no marker: scale x{scale:.3f} from the phone's height above the floor "
               f"(assumed {CAMERA_HEIGHT_M} m), about +/-10%")
        notes.append(f"no marker seen; scale from the phone held about {CAMERA_HEIGHT_M} m above the floor, "
                     "so sizes are about +/-10% (a marker on a wall in the turn makes this exact)")
    else:
        scale = 1.0
        how = "no marker correction"
        notes.append("no marker seen and no usable floor for the camera-height check; depth model scale used "
                     "uncorrected (a marker on a wall in the turn makes the room exact)")
    pts = np.vstack([c * scale + arm for c, arm in zip(clouds, arms)])
    flat = np.concatenate(flats)

    z = pts[:, 2]
    floor_z = float(np.median([-d for _, d in floors])) * scale if floors else \
        horizontal_layer(z[flat & (z < -0.3)], lowest=True)
    ups = [i for i in range(len(picked)) if kinds[i] == "up"]
    ups = [i for i in ups if i in aligned] or ups
    ceilings = [c * scale for i in ups for c in [layer(i, False)] if c]
    ceil_z = float(np.median(ceilings)) if ceilings else horizontal_layer(z[flat & (z > 0.3)], lowest=False)
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
    confidence = round(min(1.0, covered / 360) * (0.85 if scales else 0.7 if prior else 0.5)
                       * (1.0 if height else 0.7), 2)
    method = f"turn-around scan: Depth Pro metric depth on {len(picked)} frames, phone orientation, {how}"
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
