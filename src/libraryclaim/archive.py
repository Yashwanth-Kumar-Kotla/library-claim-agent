"""Keep the outputs of every run in the repository, so results survive and runs can be compared.

    results/<sweep>/<time>-<kind>/   claim_packet.json, report.html, timings.json, session.jsonl,
                                      room_scan.json, eval.md (whichever exist; never frames or video)
    results/index.md                 one line per run: counts, values, room, cost, time to packet

Only sweeps under data/sweeps are archived, so tests and scratch runs never write here.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from .config import REPO_ROOT

SWEEPS = REPO_ROOT / "data" / "sweeps"
RESULTS = REPO_ROOT / "results"
log = logging.getLogger(__name__)
FILES = ["claim_packet.json", "report.html", "eval.md", "eval.json", "ground_truth_results.xlsx",
         "out/timings.json", "out/session.jsonl", "out/room_scan.json", "out/claim_packet.live.json"]
HEADER = ("| Run | Sweep | Kind | Books | Identified | Appraisal | Items | Books replacement | Other contents "
          "| Floor m² | Model cost | Packet after sweep | Pass bars |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def summary(sweep: Path, run: str, kind: str) -> str:
    p = json.loads((sweep / "claim_packet.json").read_text())
    t, room, cur = p["totals"], p["room"], p["sweep"].get("currency", "")
    statuses = [b["status"] for b in p["books"]]
    timings = _load(sweep / "out" / "timings.json") or {}
    vision = ((timings.get("cost") or {}).get("usd") or 0.0)
    live = sum(e.get("usd", 0) for e in _jsonl(sweep / "out" / "session.jsonl") if e.get("type") == "live_usage")
    after = (timings.get("seconds") or {}).get("after_sweep")
    bars = ""
    ev = _load(sweep / "eval.json")
    if ev:
        judged = [r for r in ev if r.get("passed") is not None]
        bars = f"{sum(r['passed'] for r in judged)}/{len(judged)}"
    items = (f"{cur} {t['items_replacement_cost_low']:,.0f}–{t['items_replacement_cost_high']:,.0f}"
             if t["items_replacement_cost_high"] else "—")
    cells = [run, sweep.name, kind, t["book_count"], t["books_identified"], statuses.count("needs_appraisal"),
             len(p["items"]), f"{cur} {t['books_replacement_cost']:,.2f}", items,
             room.get("floor_area_m2") or "—", f"${vision + live:.3f}", f"{after:.1f} s" if after is not None else "—", bars]
    return "| " + " | ".join(str(c) for c in cells) + " |\n"


def archive_run(sweep: Path, kind: str, results: Path | None = None, now: datetime | None = None) -> Path | None:
    """Copy one run's outputs into results/ and add its line to the index. Returns the folder, or None
    when the sweep isn't a real one under data/sweeps, has no packet yet, or can't be archived. Never
    raises: losing the archive copy must not lose the run itself."""
    try:
        return _archive(Path(sweep).resolve(), kind, results, now)
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("results archive skipped for %s: %s", sweep, e)
        return None


def _archive(sweep: Path, kind: str, results: Path | None, now: datetime | None) -> Path | None:
    if results is None:
        if SWEEPS.resolve() not in sweep.parents:
            return None
        results = RESULTS
    if not (sweep / "claim_packet.json").is_file():
        return None
    run = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    dest = results / sweep.name / f"{run}-{kind}"
    dest.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        src = sweep / rel
        if src.is_file():
            shutil.copy2(src, dest / Path(rel).name)
    index = results / "index.md"
    if not index.is_file():
        index.write_text("# Runs\n\nEvery processed sweep, newest last. Outputs are in `results/<sweep>/<run>/`.\n\n" + HEADER)
    with index.open("a") as f:
        f.write(summary(sweep, run, kind))
    return dest


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []
