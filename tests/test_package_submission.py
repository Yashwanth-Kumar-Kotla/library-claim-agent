import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from package_submission import package  # noqa: E402


def make_sweep(root: Path) -> Path:
    sweep = root / "sweep"
    (sweep / "frames").mkdir(parents=True)
    (sweep / "out" / "crops").mkdir(parents=True)
    for name in ("00001.jpg", "00002.jpg", "00009.jpg"):
        (sweep / "frames" / name).write_bytes(b"\xff\xd8")
    (sweep / "out" / "crops" / "b1.jpg").write_bytes(b"\xff\xd8")
    packet = {"books": [{"id": "b1", "frame_ref": "frames/00001.jpg#xyxy=0,0,10,10"}],
              "items": [{"id": "i1", "frame_ref": "frames/00002.jpg"},
                        {"id": "i2", "frame_ref": "frames/00404.jpg"}]}
    (sweep / "claim_packet.json").write_text(json.dumps(packet))
    (sweep / "report.html").write_text("<html></html>")
    return sweep


def test_copies_only_referenced_frames_and_reports_missing(tmp_path):
    sweep = make_sweep(tmp_path)
    out = tmp_path / "submission"
    missing = package(sweep, out)
    assert sorted(p.name for p in (out / "frames").iterdir()) == ["00001.jpg", "00002.jpg"]
    assert (out / "out" / "crops" / "b1.jpg").exists()
    assert (out / "claim_packet.json").exists() and (out / "report.html").exists()
    assert (out / "ARCHITECTURE.md").exists() and (out / "FAILURE_LOG.md").exists()
    assert "frames/00404.jpg" in missing and "ground_truth_results.xlsx" in missing


def test_rerun_overwrites_cleanly(tmp_path):
    sweep = make_sweep(tmp_path)
    out = tmp_path / "submission"
    package(sweep, out)
    (sweep / "eval.md").write_text("scores")
    assert "eval.md" not in package(sweep, out)
    assert (out / "eval.md").read_text() == "scores"


def test_sweep_without_packet_still_copies_docs(tmp_path):
    sweep = tmp_path / "empty"
    sweep.mkdir()
    missing = package(sweep, tmp_path / "submission")
    assert "claim_packet.json" in missing
    assert (tmp_path / "submission" / "FAILURE_LOG.md").exists()
