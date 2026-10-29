"""Run the whole pipeline with no API keys, on a synthetic shelf with known answers.

    uv run python tools/demo_offline.py          # writes data/sweeps/demo-synthetic and opens the report

The shelf, the vision model and the catalogue/price responses are the same fakes the tests use
(tests/synthetic.py): the detector finds spines by colour and the reader identifies them by colour,
so everything after the model calls (tiling, merging, scale, sizes, resolution, pricing rules,
review queue, totals, report) is the real code.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from helpers import FakeWeb  # noqa: E402
from synthetic import FakeVision, build_sweep, fake_catalogue  # noqa: E402

from libraryclaim.config import Locale, Settings  # noqa: E402
from libraryclaim.pipeline import Deps, run_pipeline  # noqa: E402


def main() -> None:
    out = ROOT / "data" / "sweeps" / "demo-synthetic"
    shutil.rmtree(out, ignore_errors=True)
    build_sweep(out)
    packet = run_pipeline(out, Deps(vision=FakeVision(), get_json=FakeWeb(fake_catalogue()), ebay_token="demo"),
                          Settings(locale=Locale("US", "USD")))
    t = packet.totals
    print(f"books: {t.book_count} ({t.books_identified} identified, {t.books_unidentified} unidentified)")
    print(f"books replacement: {t.books_replacement_cost}  used: {t.books_used_value}  excluded: {t.excluded_from_totals}")
    print(f"review queue: {len(packet.review_queue)} lines")
    print(f"packet: {out / 'claim_packet.json'}")
    print(f"report: {out / 'report.html'}")
    if sys.platform == "darwin" and "--no-open" not in sys.argv:
        subprocess.run(["open", str(out / "report.html")], check=False)


if __name__ == "__main__":
    main()
