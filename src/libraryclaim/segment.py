"""One image per shelf row.

The live agent walks the user through the shelves row by row, so the frames of each row are
known. Stitching them into a panorama means each spine appears exactly once, which is what
makes the count honest. If stitching fails (too little overlap, too much parallax) the row
falls back to its sharpest single frame and says so, rather than double counting.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .quality import to_work_gray


@dataclass
class RowImage:
    image: np.ndarray
    method: str            # single_frame | stitched | sharpest_frame_fallback
    source_frames: list[str]
    note: str = ""


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(to_work_gray(img), cv2.CV_64F).var())


def row_image(frames: list[tuple[str, np.ndarray]]) -> RowImage:
    """frames: (ref, image) in capture order."""
    frames = [(ref, img) for ref, img in frames if img is not None and img.size]
    if not frames:
        raise ValueError("row has no usable frames")
    if len(frames) == 1:
        return RowImage(frames[0][1], "single_frame", [frames[0][0]])
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch([img for _, img in frames])
    if status == cv2.Stitcher_OK and pano is not None and pano.size:
        return RowImage(pano, "stitched", [ref for ref, _ in frames])
    ref, img = max(frames, key=lambda f: sharpness(f[1]))
    return RowImage(img, "sharpest_frame_fallback", [ref],
                    note=f"could not stitch {len(frames)} frames (status {status}); counted from one frame, may undercount")
