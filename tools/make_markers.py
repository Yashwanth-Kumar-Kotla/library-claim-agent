"""Print-ready ArUco marker sheets at exact physical size.

Usage: uv run python tools/make_markers.py --out ground_truth/markers.pdf

Each A4/Letter page holds one marker with a printed ID and side length. After printing,
measure the black square with a ruler; if it is not exactly --size-cm, pass the measured
value to the pipeline (MARKER_SIZE_CM) rather than trusting the printer.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DICT = cv2.aruco.DICT_4X4_50
DPI = 300
PAGES = {"letter": (21.59, 27.94), "a4": (21.0, 29.7)}


def cm_to_px(cm: float) -> int:
    return round(cm / 2.54 * DPI)


def marker_page(marker_id: int, size_cm: float, page: str) -> Image.Image:
    pw, ph = (cm_to_px(v) for v in PAGES[page])
    side = cm_to_px(size_cm)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    canvas = Image.new("L", (pw, ph), 255)
    x, y = (pw - side) // 2, (ph - side) // 2
    canvas.paste(Image.fromarray(marker), (x, y))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=48)
    label = f"ArUco 4x4_50  id={marker_id}  side={size_cm:.1f} cm  (print at 100%, no scaling)"
    draw.text((x, y + side + 60), label, fill=0, font=font)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ground_truth/markers.pdf")
    ap.add_argument("--ids", default="0,1,2,3")
    ap.add_argument("--size-cm", type=float, default=15.0)
    ap.add_argument("--page", choices=PAGES, default="letter")
    args = ap.parse_args()
    pages = [marker_page(int(i), args.size_cm, args.page) for i in args.ids.split(",")]
    pages[0].save(args.out, save_all=True, append_images=pages[1:], resolution=DPI)
    print(f"wrote {len(pages)} pages to {args.out}")


if __name__ == "__main__":
    main()
