"""Score a claim packet against hand-collected ground truth, one line per pass bar.

    uv run python -m libraryclaim.evaluate data/sweeps/<id> ground_truth/ground_truth.xlsx

Scored on every shelf in the ground truth, never a subset. Where the brief leaves a metric's
exact definition open, the choice is written next to the number.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from openpyxl import load_workbook
from rapidfuzz import fuzz

from .archive import archive_run
from .resolve import main_title, norm
from .schema import ClaimPacket

TITLE_MATCH = 90


@dataclass
class Result:
    measure: str
    value: str
    bar: str
    passed: bool | None  # None = not enough ground truth to judge
    detail: str = ""


def rows(ws) -> list[dict]:
    """Sheet rows below the note line and header, as dicts; blank rows skipped."""
    values = list(ws.iter_rows(values_only=True))
    if len(values) < 2:
        return []
    header = [str(h).strip() if h is not None else "" for h in values[1]]
    out = []
    for v in values[2:]:
        if v is None or all(c in (None, "") for c in v):
            continue
        out.append({h: c for h, c in zip(header, v) if h})
    return out


def load_truth(path: Path) -> dict:
    wb = load_workbook(path, data_only=True)
    summary = {r[0]: r[1] for r in wb["summary"].iter_rows(values_only=True) if r and r[0]}
    return {
        "summary": summary,
        "books": rows(wb["books"]),
        "spines": rows(wb["spine_measurements"]),
        "prices": rows(wb["prices"]),
        "items": rows(wb["items"]),
        "room": {r["what"]: r for r in rows(wb["room"]) if r.get("what")},
    }


def same_title(a: str, b: str) -> bool:
    a, b = norm(main_title(a)), norm(main_title(b))
    return bool(a and b) and fuzz.ratio(a, b) >= TITLE_MATCH


def pct_err(got, want) -> float | None:
    try:
        got, want = float(got), float(want)
    except (TypeError, ValueError):
        return None
    return abs(got - want) / want if want else None


def yes(v) -> bool:
    return str(v).strip().lower() in {"y", "yes", "true", "1"}


def score(packet: ClaimPacket, truth: dict, timings: dict | None = None) -> list[Result]:
    out = []
    books = packet.books
    identified = [b for b in books if b.status in ("identified", "needs_appraisal") and b.title]

    true_count = truth["summary"].get("true_book_count") or len(truth["books"])
    if true_count:
        e = abs(len(books) - true_count) / true_count
        out.append(Result("Book count", f"{len(books)} vs {true_count} ({e:.1%} off)", "within 5%", e <= 0.05))
    else:
        out.append(Result("Book count", "no ground truth", "within 5%", None))

    gt_titles = [r["title"] for r in truth["books"] if r.get("title")]
    legible = [r["title"] for r in truth["books"] if r.get("title") and yes(r.get("legible", "y"))]
    if legible:
        found = sum(any(same_title(b.title, t) for b in identified) for t in legible)
        wrong = [b for b in identified if not any(same_title(b.title, t) for t in gt_titles)]
        rate = found / len(legible)
        wrong_rate = len(wrong) / len(identified) if identified else 0.0
        out.append(Result("Title identification", f"{found}/{len(legible)} legible spines correct ({rate:.0%})",
                          "at least 70%", rate >= 0.70))
        out.append(Result("Confidently wrong", f"{len(wrong)}/{len(identified)} identified titles not on the shelf ({wrong_rate:.1%})",
                          "no more than 3%", wrong_rate <= 0.03,
                          "denominator: books the system marked identified; " + ", ".join(b.title for b in wrong[:5])))
    else:
        out.append(Result("Title identification", "no ground truth", "at least 70%", None))

    errs, missing = [], 0
    for r in truth["spines"]:
        b = next((b for b in identified if r.get("title") and same_title(b.title, r["title"])), None)
        if b is None:
            b = next((b for b in books if b.shelf == f"{r.get('shelf_unit', '')}{r.get('row', '')}"
                      and b.position == r.get("position")), None)
        if b is None or b.spine_height_cm is None:
            missing += 1
            continue
        for got, want in ((b.spine_height_cm, r.get("height_cm")), (b.spine_thickness_cm, r.get("thickness_cm"))):
            e = pct_err(got, want)
            if e is not None:
                errs.append(e)
    out.append(dim_result("Spine dimensions", errs, missing, len(truth["spines"]), 0.15))

    errs, missing = [], 0
    for r in truth["prices"]:
        b = next((b for b in books if (r.get("isbn") and b.isbn == str(r["isbn"]))
                  or (r.get("title") and same_title(b.title, r["title"]))), None)
        pairs = [] if b is None else [(b.replacement_cost.amount, r.get("replacement_new")), (b.used_value.amount, r.get("used_value"))]
        got_any = False
        for got, want in pairs:
            if got is not None and want not in (None, ""):
                e = pct_err(got, want)
                if e is not None:
                    errs.append(e)
                    got_any = True
        missing += not got_any
    out.append(dim_result("Book prices", errs, missing, len(truth["prices"]), 0.25))

    gt_items = truth["items"]
    if gt_items:
        pool = list(packet.items)
        hit = 0
        for r in gt_items:
            cat = str(r.get("category", "")).strip().lower()
            match = max((i for i in pool if i.category == cat),
                        key=lambda i: fuzz.token_set_ratio(norm(i.description), norm(str(r.get("item", "")))), default=None)
            if match:
                pool.remove(match)
                hit += 1
        rate = hit / len(gt_items)
        out.append(Result("Non-book items", f"{hit}/{len(gt_items)} found with correct category ({rate:.0%})", "at least 80%", rate >= 0.80))
    else:
        out.append(Result("Non-book items", "no ground truth", "at least 80%", None))

    floor = truth["room"].get("room_floor", {})
    ceiling = truth["room"].get("ceiling", {})
    L, W, H = floor.get("length_m"), floor.get("width_m"), ceiling.get("height_m")
    if L and W:
        e = pct_err(packet.room.floor_area_m2, L * W)
        out.append(Result("Floor area", f"{packet.room.floor_area_m2} vs {L * W:.2f} m2" + (f" ({e:.1%} off)" if e is not None else ""),
                          "within 10%", None if e is None else e <= 0.10))
    else:
        out.append(Result("Floor area", "no ground truth", "within 10%", None))
    if L and W and H:
        want = 2 * (L + W) * H
        e = pct_err(packet.room.wall_area_m2, want)
        out.append(Result("Wall area", f"{packet.room.wall_area_m2} vs {want:.2f} m2" + (f" ({e:.1%} off)" if e is not None else ""),
                          "within 15%", None if e is None else e <= 0.15, "gross wall area, rectangular room from tape"))
    else:
        out.append(Result("Wall area", "no ground truth", "within 15%", None))

    total = (timings or {}).get("seconds", {}).get("total")
    out.append(Result("Time to packet", f"{total:.0f} s" if total is not None else "not recorded", "under 5 minutes",
                      None if total is None else total < 300))
    return out


def dim_result(name: str, errs: list[float], missing: int, n: int, bar: float) -> Result:
    if not n:
        return Result(name, "no ground truth", f"within {bar:.0%}", None)
    if not errs:
        return Result(name, f"0 of {n} sample lines measured by the system", f"within {bar:.0%}", False)
    mape = statistics.mean(errs)
    within = sum(e <= bar for e in errs) / len(errs)
    return Result(name, f"mean error {mape:.1%}; {within:.0%} of values within {bar:.0%}; {missing}/{n} not produced",
                  f"within {bar:.0%}", mape <= bar and missing == 0,
                  "pass = mean absolute % error within the bar AND every sample line produced")


def as_markdown(results: list[Result]) -> str:
    mark = {True: "PASS", False: "FAIL", None: "n/a"}
    lines = ["| Measure | System vs truth | Pass bar | Result |", "|---|---|---|---|"]
    lines += [f"| {r.measure} | {r.value} | {r.bar} | {mark[r.passed]} |" for r in results]
    return "\n".join(lines)


def write_workbook(truth_path: Path, out_path: Path, results: list[Result], packet: ClaimPacket, truth: dict) -> None:
    """A copy of the ground truth workbook with the system's numbers next to it: a results sheet
    (every pass bar) and each hand-read title beside what the system said. The original is untouched."""
    from openpyxl.styles import Font, PatternFill

    wb = load_workbook(truth_path)
    for name in ("results", "titles_compared"):
        if name in wb.sheetnames:
            del wb[name]
    ws = wb.create_sheet("results", 0)
    ws.append(["Measure", "System vs truth", "Pass bar", "Result", "How it was scored"])
    fills = {True: "D7F0DC", False: "F8D7DA", None: "EEEEEE"}
    for r in results:
        ws.append([r.measure, r.value, r.bar, {True: "PASS", False: "FAIL", None: "n/a"}[r.passed], r.detail])
        ws.cell(ws.max_row, 4).fill = PatternFill("solid", fgColor=fills[r.passed])
    for cell in ws[1]:
        cell.font = Font(bold=True)

    ws = wb.create_sheet("titles_compared", 1)
    ws.append(["shelf_unit", "row", "position", "legible", "true title", "system line", "system status", "system title", "match"])
    used = set()
    for t in truth["books"]:
        title = t.get("title") or ""
        hit = next((b for b in packet.books if b.id not in used and b.title and title and same_title(b.title, title)), None)
        if hit:
            used.add(hit.id)
        ws.append([t.get("shelf_unit"), t.get("row"), t.get("position"), t.get("legible"), title,
                   hit.id if hit else "", hit.status if hit else "", hit.title if hit else "",
                   "yes" if hit else ("not found" if yes(t.get("legible", "y")) else "not legible")])
    for b in packet.books:
        if b.title and b.id not in used and b.status != "unidentified":
            ws.append(["", "", "", "", "", b.id, b.status, b.title, "NOT ON SHELF (confidently wrong)"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    wb.save(out_path)


def main(argv=None) -> None:
    sweep, truth_path = (Path(a) for a in (argv or sys.argv[1:])[:2])
    packet = ClaimPacket.model_validate_json((sweep / "claim_packet.json").read_text())
    timings_path = sweep / "out" / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.is_file() else None
    truth = load_truth(truth_path)
    results = score(packet, truth, timings)
    (sweep / "eval.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    md = as_markdown(results)
    (sweep / "eval.md").write_text(md + "\n")
    write_workbook(truth_path, sweep / "ground_truth_results.xlsx", results, packet, truth)
    print(md)
    print(f"\nworkbook with results: {sweep / 'ground_truth_results.xlsx'}")
    archive_run(sweep, "eval")


if __name__ == "__main__":
    main()
