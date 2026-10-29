"""Metric scale from a printed ArUco marker taped to the shelf front.

The four marker corners give a homography from image pixels to centimetres on the plane
the marker sits on. Spines sit on (nearly) that same plane, so projecting a spine's box
through it gives real height and thickness even when the phone is held at an angle.
A single cm-per-pixel factor would not survive perspective; the homography does.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MIN_MARKER_PX = 40  # below this the corner error is a large share of the side length

# Book heights outside this range are almost certainly a measurement error, not a book.
PLAUSIBLE_HEIGHT_CM = (9.0, 50.0)
PLAUSIBLE_THICKNESS_CM = (0.15, 12.0)


@dataclass
class Marker:
    id: int
    corners: np.ndarray  # (4, 2) px, clockwise from top-left

    @property
    def side_px(self) -> float:
        c = self.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


@dataclass
class PlaneScale:
    H: np.ndarray        # 3x3, image px -> plane cm
    marker_id: int
    marker_side_px: float
    method: str = "aruco_marker_homography"

    def to_cm(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    @property
    def reliable(self) -> bool:
        return self.marker_side_px >= MIN_MARKER_PX


def _params() -> cv2.aruco.DetectorParameters:
    p = cv2.aruco.DetectorParameters()
    # Default drops markers whose contour comes within 3 px of the frame edge; in a sweep the
    # marker is often at the edge. Measured: 19/30 -> 30/30 detections on textured shelves.
    p.minDistanceToBorder = 0
    # Corner error is measurement error, so refine corners to sub-pixel.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), _params())


def find_markers(img: np.ndarray) -> list[Marker]:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return []
    return [Marker(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]


def plane_from_marker(marker: Marker, size_cm: float) -> PlaneScale:
    if size_cm <= 0:
        raise ValueError("marker size must be positive")
    square = np.array([[0, 0], [size_cm, 0], [size_cm, size_cm], [0, size_cm]], np.float64)
    H = cv2.getPerspectiveTransform(marker.corners.astype(np.float32), square.astype(np.float32))
    return PlaneScale(H.astype(np.float64), marker.id, marker.side_px)


def box_size_cm(scale: PlaneScale, box: tuple[float, float, float, float]) -> tuple[float, float]:
    """(width_cm, height_cm) of an axis-aligned pixel box (x0, y0, x1, y1), averaging opposite edges."""
    x0, y0, x1, y1 = box
    tl, tr, br, bl = scale.to_cm(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(width), float(height)


@dataclass
class SpineSize:
    height_cm: float
    thickness_cm: float
    lying_flat: bool
    plausible: bool


def spine_size(scale: PlaneScale, box: tuple[float, float, float, float], lying_flat: bool | None = None) -> SpineSize:
    """A spine's long side is its height. Upright books are tall boxes, flat ones are wide.
    The detector's orientation label wins when given; otherwise the box shape decides."""
    w, h = box_size_cm(scale, box)
    flat = (w > h) if lying_flat is None else lying_flat
    height, thickness = (w, h) if flat else (h, w)
    ok = (PLAUSIBLE_HEIGHT_CM[0] <= height <= PLAUSIBLE_HEIGHT_CM[1]
          and PLAUSIBLE_THICKNESS_CM[0] <= thickness <= PLAUSIBLE_THICKNESS_CM[1])
    return SpineSize(round(height, 1), round(thickness, 2), flat, ok)


def shelf_run_cm(scale: PlaneScale, boxes: list[tuple[float, float, float, float]]) -> float:
    """Length of shelf occupied by books: union of their horizontal extents on the shelf plane,
    so a stack of flat books counts once and gaps between books don't count."""
    spans = []
    for x0, y0, x1, y1 in boxes:
        ymid = (y0 + y1) / 2
        a, b = scale.to_cm(np.array([[x0, ymid], [x1, ymid]]))[:, 0]
        spans.append((min(a, b), max(a, b)))
    total, end = 0.0, -np.inf
    for a, b in sorted(spans):
        if a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return float(total)
