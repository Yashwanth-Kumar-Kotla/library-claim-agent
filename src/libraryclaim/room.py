"""Room figures from a floor outline and a ceiling height.

Whatever produced the outline (RoomPlan on LiDAR, a marker-scaled reconstruction, or
per-wall marker measurements), this module only does geometry, so it is the same for every
capture path and fully testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

M2_TO_FT2 = 10.7639
RIGHT_ANGLE_TOL_DEG = 8.0


@dataclass
class Opening:
    width_m: float
    height_m: float


@dataclass
class RoomFigures:
    length_m: float
    width_m: float
    height_m: float
    floor_area_m2: float
    wall_area_m2: float           # gross: perimeter x height
    wall_area_net_m2: float       # minus doors and windows we know about
    shelved_wall_area_m2: float
    shape_note: str
    warnings: list[str]

    def as_room_fields(self) -> dict:
        return {
            "length_m": round(self.length_m, 2),
            "width_m": round(self.width_m, 2),
            "height_m": round(self.height_m, 2),
            "floor_area_m2": round(self.floor_area_m2, 2),
            "wall_area_m2": round(self.wall_area_m2, 2),
            "shelved_wall_area_m2": round(self.shelved_wall_area_m2, 2),
            "floor_area_ft2": round(self.floor_area_m2 * M2_TO_FT2, 1),
            "wall_area_ft2": round(self.wall_area_m2 * M2_TO_FT2, 1),
            "shape_note": self.shape_note,
        }


def clean_outline(corners) -> np.ndarray:
    pts = np.asarray(corners, np.float64).reshape(-1, 2)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]  # closed polygon passed in
    keep = [pts[0]] if len(pts) else []
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > 1e-3:
            keep.append(p)
    pts = np.array(keep)
    if len(pts) >= 3:
        # Drop collinear points: a wall reported in two pieces is still one wall.
        out = []
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            u, v = b - a, c - b
            if abs(u[0] * v[1] - u[1] * v[0]) > 1e-6:
                out.append(b)
        pts = np.array(out)
    return pts


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def self_intersects(pts: np.ndarray) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # neighbours share a vertex
            p1, p2 = edges[i]
            q1, q2 = edges[j]
            if (cross(p1, p2, q1) * cross(p1, p2, q2) < 0) and (cross(q1, q2, p1) * cross(q1, q2, p2) < 0):
                return True
    return False


def is_rectangle(pts: np.ndarray) -> bool:
    if len(pts) != 4:
        return False
    for i in range(4):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 4]
        u, v = a - b, c - b
        cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        if abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 90) > RIGHT_ANGLE_TOL_DEG:
            return False
    return True


def room_figures(corners, height_m: float, openings: list[Opening] = (),
                 shelving: list[tuple[float, float]] = ()) -> RoomFigures:
    """corners: floor outline in metres, in order around the room. shelving: (width_m, height_m)
    of each shelving unit's front, which is the wall area it covers."""
    if not height_m or height_m <= 0:
        raise ValueError("ceiling height must be positive")
    pts = clean_outline(corners)
    if len(pts) < 3:
        raise ValueError("need at least 3 distinct, non-collinear corners")
    if self_intersects(pts):
        raise ValueError("floor outline crosses itself; corners are out of order")

    warnings = []
    area = polygon_area(pts)
    perimeter = float(sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(len(pts))))
    if height_m < 1.8 or height_m > 6:
        warnings.append(f"ceiling height {height_m:.2f} m is unusual for a room")

    if is_rectangle(pts):
        sides = sorted([np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])], reverse=True)
        length, width = sides
        note = "rectangular"
    else:
        (_, _), (a, b), _ = cv2.minAreaRect(pts.astype(np.float32))
        length, width = max(a, b), min(a, b)
        note = (f"non-rectangular, {len(pts)} corners; floor area from the outline polygon, "
                f"length/width are the smallest enclosing rectangle")

    gross_wall = perimeter * height_m
    opening_area = sum(o.width_m * o.height_m for o in openings)
    if opening_area > gross_wall:
        warnings.append("openings exceed wall area; ignored")
        opening_area = 0.0
    shelved = sum(w * h for w, h in shelving)
    if shelved > gross_wall - opening_area:
        warnings.append("shelving area exceeds available wall area; check shelving measurements")
    for w, h in shelving:
        if h > height_m + 0.05:
            warnings.append(f"a shelving unit ({h:.2f} m) is taller than the ceiling ({height_m:.2f} m)")

    return RoomFigures(float(length), float(width), float(height_m), area, gross_wall,
                       gross_wall - opening_area, shelved, note, warnings)
