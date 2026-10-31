"""Collect the submission for one sweep: packet, report, the frames it references, ground truth results, docs.

Usage: uv run python tools/package_submission.py data/sweeps/<sweep> [--out submission]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ["claim_packet.json", "report.html", "eval.md", "eval.json", "ground_truth_results.xlsx",
           "out/timings.json", "out/session.jsonl", "out/room_scan.json"]
DOCS = ["ARCHITECTURE.md", "FAILURE_LOG.md"]
FRAME_REF = re.compile(r'"(frames/[^"#]+)')


def package(sweep: Path, out: Path) -> list[str]:
    """Copy everything a reviewer needs and return the list of missing outputs."""
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in OUTPUTS:
        src = sweep / name
        if src.exists():
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / name)
        else:
            missing.append(name)
    # The report shows a crop per book and links each line to the frame it was seen in.
    if (sweep / "out" / "crops").is_dir():
        shutil.copytree(sweep / "out" / "crops", out / "out" / "crops", dirs_exist_ok=True)
    packet = (sweep / "claim_packet.json").read_text() if (sweep / "claim_packet.json").exists() else ""
    for ref in sorted(set(FRAME_REF.findall(packet))):
        if (sweep / ref).exists():
            (out / ref).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sweep / ref, out / ref)
        else:
            missing.append(ref)
    for doc in DOCS:
        shutil.copy2(ROOT / doc, out / doc)
    return missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "submission")
    args = ap.parse_args()
    missing = package(args.sweep, args.out)
    frames = len(list((args.out / "frames").glob("*"))) if (args.out / "frames").is_dir() else 0
    print(f"submission written to {args.out} ({frames} frames)")
    for name in missing:
        print(f"  missing: {name}")


if __name__ == "__main__":
    main()
