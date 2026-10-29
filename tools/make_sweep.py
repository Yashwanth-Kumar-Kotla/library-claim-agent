"""Turn ordinary phone photos into a sweep folder the pipeline can run on.

For testing before the live app exists: photograph each shelf row (marker taped to the shelf
and in the shot), AirDrop the photos to the Mac, then

    uv run python tools/make_sweep.py data/sweeps/home-1 \\
        --row A1 ~/Downloads/IMG_0101.HEIC \\
        --row A2 ~/Downloads/IMG_0102.HEIC ~/Downloads/IMG_0103.HEIC \\
        --room ~/Downloads/IMG_0110.HEIC ~/Downloads/IMG_0111.HEIC

Several photos for one row are stitched into a panorama, so overlap them by about a third.
HEIC is converted with macOS `sips`. Large photos are scaled so the long side is at most 4000 px.
No room geometry is written: that has to come from the sweep, not from typing in measurements.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2

MAX_SIDE = 4000


def load(path: Path):
    if path.suffix.lower() in (".heic", ".heif"):
        tmp = Path(tempfile.mkdtemp()) / (path.stem + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)], check=True, capture_output=True)
        path = tmp
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"can't read image {path}")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--row", nargs="+", action="append", default=[], metavar=("SHELF", "PHOTO"),
                    help="shelf name (e.g. A1) followed by its photos, left to right")
    ap.add_argument("--room", nargs="*", default=[], help="photos of the room for non-book items")
    ap.add_argument("--country", default="US")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--device", default="iPhone (photos)")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; pick a new sweep name")
    frames = args.out / "frames"
    frames.mkdir(parents=True)
    n = 0

    def save(photo: str) -> str:
        nonlocal n
        n += 1
        ref = f"frames/{n:04d}.jpg"
        cv2.imwrite(str(args.out / ref), load(Path(photo).expanduser()), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return ref

    try:
        rows = []
        for shelf, *photos in args.row:
            if not photos:
                raise SystemExit(f"--row {shelf} has no photos")
            rows.append({"shelf": shelf, "frames": [save(p) for p in photos]})
        room = [save(p) for p in args.room]
    except BaseException:
        shutil.rmtree(args.out, ignore_errors=True)
        raise

    (args.out / "rows.json").write_text(json.dumps(rows, indent=2))
    (args.out / "room_frames.json").write_text(json.dumps(room, indent=2))
    (args.out / "sweep.json").write_text(json.dumps({
        "id": args.out.name, "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": args.device, "duration_s": 0, "country": args.country.upper(), "currency": args.currency.upper(),
        "capture_path": "browser",
    }, indent=2))
    print(f"wrote {args.out}: {len(rows)} shelf rows, {len(room)} room photos")
    print(f"next: uv run python -m libraryclaim run {args.out}")


if __name__ == "__main__":
    main()
