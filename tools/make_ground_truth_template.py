"""Blank ground truth workbook. Fill it by hand before tuning anything.

Usage: uv run python tools/make_ground_truth_template.py --out ground_truth/ground_truth.xlsx
"""

from __future__ import annotations

import argparse

from openpyxl import Workbook
from openpyxl.styles import Font

SHEETS = {
    "books": (
        "One row per book in shelf order. legible = could a person read the spine from where the camera stands (y/n).",
        ["shelf_unit", "row", "position", "legible", "title", "author", "publisher", "lying_flat", "notes"],
    ),
    "spine_measurements": (
        "20 spines with a ruler, spread across tall/thin/flat. position refers to the books sheet.",
        ["shelf_unit", "row", "position", "title", "height_cm", "thickness_cm"],
    ),
    "prices": (
        "15 books priced by hand from the same source types the system uses (Google Books new, eBay used).",
        ["title", "isbn", "source", "url", "retrieved_at", "currency", "replacement_new", "used_value", "condition"],
    ),
    "items": (
        "Every non-book item in the room (at least 8).",
        ["item", "category", "brand_model", "w_cm", "h_cm", "d_cm", "notes"],
    ),
    "room": (
        "Tape measurements. Add one row per wall opening (door/window) and per shelving unit.",
        ["what", "length_m", "width_m", "height_m", "notes"],
    ),
}

ROOM_ROWS = [
    ["room_floor", None, None, None, "length = longest wall"],
    ["ceiling", None, None, None, "height only"],
    ["door_1", None, None, None, "width x height"],
    ["window_1", None, None, None, "width x height"],
    ["shelving_unit_A", None, None, None, "width x height x depth"],
    ["shelving_unit_B", None, None, None, "width x height x depth"],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ground_truth/ground_truth.xlsx")
    args = ap.parse_args()
    wb = Workbook()
    wb.remove(wb.active)
    for name, (note, headers) in SHEETS.items():
        ws = wb.create_sheet(name)
        ws.append([note])
        ws["A1"].font = Font(italic=True)
        ws.append(headers)
        for cell in ws[2]:
            cell.font = Font(bold=True)
        if name == "room":
            for row in ROOM_ROWS:
                ws.append(row)
    summary = wb.create_sheet("summary", 0)
    summary.append(["true_book_count", None])
    summary.append(["shelving_units", None])
    summary.append(["country", "US"])
    summary.append(["currency", "USD"])
    wb.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
