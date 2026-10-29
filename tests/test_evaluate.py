import subprocess
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook

from libraryclaim.evaluate import as_markdown, load_truth, score
from libraryclaim.schema import Book, BookPrice, ClaimPacket, Item, Room

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def truth_file(tmp_path):
    path = tmp_path / "gt.xlsx"
    subprocess.run([sys.executable, str(ROOT / "tools/make_ground_truth_template.py"), "--out", str(path)], check=True)
    wb = load_workbook(path)
    wb["summary"]["B1"] = 4
    for r in [["A", 1, 1, "y", "Dune", "Frank Herbert"], ["A", 1, 2, "y", "The Hobbit", "Tolkien"],
              ["A", 1, 3, "n", "Faded Thing", ""], ["A", 1, 4, "y", "Sapiens", "Harari"]]:
        wb["books"].append(r)
    wb["spine_measurements"].append(["A", 1, 1, "Dune", 17.8, 3.2])
    wb["spine_measurements"].append(["A", 1, 2, "The Hobbit", 19.8, 2.2])
    wb["prices"].append(["Dune", "", "eBay", "u", "d", "USD", 20, 8])
    wb["items"].append(["floor lamp", "lighting"])
    wb["items"].append(["portrait", "art"])
    ws = wb["room"]
    ws["B3"], ws["C3"] = 4.0, 3.0      # room_floor length, width
    ws["D4"] = 2.5                     # ceiling height
    wb.save(path)
    return path


def packet(**over):
    books = [
        Book(id="A1-01", shelf="A1", position=1, status="identified", title="Dune", spine_height_cm=18.0, spine_thickness_cm=3.0,
             replacement_cost=BookPrice(amount=21), used_value=BookPrice(amount=8.5)),
        Book(id="A1-02", shelf="A1", position=2, status="identified", title="The Hobbit: or There and Back Again",
             spine_height_cm=20.5, spine_thickness_cm=2.1),
        Book(id="A1-03", shelf="A1", position=3, status="unidentified"),
        Book(id="A1-04", shelf="A1", position=4, status="unidentified"),
    ]
    p = ClaimPacket(books=books, items=[Item(id="I-01", category="lighting", description="brass floor lamp")],
                    room=Room(floor_area_m2=12.4, wall_area_m2=36.0))
    return p.model_copy(update=over)


def by_name(results):
    return {r.measure: r for r in results}


def test_scores_each_bar(truth_file):
    r = by_name(score(packet(), load_truth(truth_file), {"seconds": {"total": 140}}))
    assert r["Book count"].passed
    assert "2/3" in r["Title identification"].value and not r["Title identification"].passed
    assert r["Confidently wrong"].passed
    assert r["Spine dimensions"].passed
    assert r["Book prices"].passed
    assert "1/2" in r["Non-book items"].value and not r["Non-book items"].passed
    assert r["Floor area"].passed and r["Wall area"].passed
    assert r["Time to packet"].passed


def test_confidently_wrong_counts_titles_not_on_shelf(truth_file):
    p = packet()
    p.books[2].status, p.books[2].title = "identified", "Twilight"
    r = by_name(score(p, load_truth(truth_file)))
    assert not r["Confidently wrong"].passed and "Twilight" in r["Confidently wrong"].detail


def test_missing_sample_lines_fail_the_bar(truth_file):
    p = packet()
    p.books[1].spine_height_cm = None
    r = by_name(score(p, load_truth(truth_file)))
    assert not r["Spine dimensions"].passed and "1/2 not produced" in r["Spine dimensions"].value


def test_wrong_count_and_room(truth_file):
    p = packet(books=packet().books[:3], room=Room(floor_area_m2=15, wall_area_m2=None))
    r = by_name(score(p, load_truth(truth_file)))
    assert not r["Book count"].passed and not r["Floor area"].passed
    assert r["Wall area"].passed is None


def test_empty_ground_truth_reports_na_not_pass(tmp_path):
    path = tmp_path / "empty.xlsx"
    subprocess.run([sys.executable, str(ROOT / "tools/make_ground_truth_template.py"), "--out", str(path)], check=True)
    results = score(ClaimPacket(), load_truth(path))
    assert all(r.passed is not True for r in results)
    assert "| Book count |" in as_markdown(results)
