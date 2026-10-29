import json
from datetime import datetime

import pytest

from libraryclaim import archive
from libraryclaim.archive import archive_run
from libraryclaim.packet import finalize
from libraryclaim.schema import Book, BookPrice, ClaimPacket, Item, ItemPrice

GOOD = dict(source="eBay US", url="https://www.ebay.com/sch/x", retrieved_at="2026-10-28T20:00:00+00:00")
T1 = datetime(2026, 10, 29, 9, 0, 0)


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "SWEEPS", tmp_path / "data" / "sweeps")
    root = tmp_path / "data" / "sweeps" / "live-1"
    (root / "frames").mkdir(parents=True)
    (root / "frames" / "f1.jpg").write_bytes(b"\xff\xd8")
    (root / "out").mkdir()
    p = ClaimPacket(books=[Book(id="b1", status="identified", title="Dune", frame_ref="frames/f1.jpg",
                                replacement_cost=BookPrice(amount=20, **GOOD)),
                           Book(id="b2", status="needs_appraisal", frame_ref="frames/f1.jpg")],
                    items=[Item(id="i1", status="range", frame_ref="frames/f1.jpg",
                                replacement_cost=ItemPrice(low=40, high=60, **GOOD))])
    (root / "claim_packet.json").write_text(finalize(p, root, None).model_dump_json())
    (root / "report.html").write_text("<html></html>")
    return root


def rows(results):
    return [l for l in (results / "index.md").read_text().splitlines() if l.startswith("| 2026")]


def test_copies_outputs_but_never_frames(sweep, tmp_path):
    (sweep / "out" / "timings.json").write_text(json.dumps({"seconds": {"after_sweep": 4.25}, "cost": {"usd": 0.05}}))
    dest = archive_run(sweep, "replay", tmp_path / "results", T1)
    assert dest == tmp_path / "results" / "live-1" / "20261029-090000-replay"
    assert sorted(f.name for f in dest.iterdir()) == ["claim_packet.json", "report.html", "timings.json"]
    assert not any(p.suffix == ".jpg" for p in (tmp_path / "results").rglob("*"))


def test_index_line_has_counts_values_cost_and_latency(sweep, tmp_path):
    (sweep / "out" / "timings.json").write_text(json.dumps({"seconds": {"after_sweep": 4.25}, "cost": {"usd": 0.05}}))
    (sweep / "out" / "session.jsonl").write_text(
        json.dumps({"type": "live_usage", "usd": 0.10}) + "\n\n" + json.dumps({"type": "agent", "text": "hi"}) + "\n")
    archive_run(sweep, "live", tmp_path / "results", T1)
    [line] = rows(tmp_path / "results")
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[:7] == ["20261029-090000", "live-1", "live", "2", "1", "1", "1"]
    assert "20.00" in cells[7] and "40–60" in cells[8]
    assert cells[9] == "—"  # room not measured
    assert cells[10] == "$0.150" and cells[11] == "4.2 s"


def test_runs_append_and_header_written_once(sweep, tmp_path):
    archive_run(sweep, "replay", tmp_path / "results", T1)
    archive_run(sweep, "replay", tmp_path / "results", datetime(2026, 10, 29, 9, 5, 0))
    text = (tmp_path / "results" / "index.md").read_text()
    assert text.count("| Run |") == 1 and len(rows(tmp_path / "results")) == 2


def test_missing_timings_and_bad_session_log_still_archive(sweep, tmp_path):
    (sweep / "out" / "session.jsonl").write_text("{not json\n")
    archive_run(sweep, "live", tmp_path / "results", T1)
    cells = [c.strip() for c in rows(tmp_path / "results")[0].strip("|").split("|")]
    assert cells[10] == "$0.000" and cells[11] == "—"


def test_eval_pass_count(sweep, tmp_path):
    (sweep / "eval.json").write_text(json.dumps([{"passed": True}, {"passed": False}, {"passed": None},
                                                 {"passed": True}]))
    archive_run(sweep, "eval", tmp_path / "results", T1)
    assert rows(tmp_path / "results")[0].rstrip().endswith("| 2/3 |")


def test_no_packet_means_nothing_written(sweep, tmp_path):
    (sweep / "claim_packet.json").unlink()
    assert archive_run(sweep, "live", tmp_path / "results", T1) is None
    assert not (tmp_path / "results").exists()


def test_corrupt_packet_does_not_raise(sweep, tmp_path):
    (sweep / "claim_packet.json").write_text("{}")
    assert archive_run(sweep, "live", tmp_path / "results", T1) is None


def test_default_archive_ignores_sweeps_outside_data(tmp_path, monkeypatch):
    results = tmp_path / "repo_results"
    monkeypatch.setattr(archive, "RESULTS", results)
    stray = tmp_path / "scratch"
    stray.mkdir()
    (stray / "claim_packet.json").write_text("{}")
    assert archive_run(stray, "replay") is None
    assert not results.exists()
