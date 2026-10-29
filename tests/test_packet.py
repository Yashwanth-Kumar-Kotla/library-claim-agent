import pytest

from libraryclaim.packet import finalize
from libraryclaim.report import render
from libraryclaim.schema import Book, BookPrice, ClaimPacket, Item, ItemPrice

GOOD = dict(source="eBay US", url="https://www.ebay.com/sch/x", retrieved_at="2026-10-28T20:00:00+00:00")


@pytest.fixture
def root(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "f1.jpg").write_bytes(b"\xff\xd8")
    return tmp_path


def test_price_without_evidence_is_removed(root):
    p = ClaimPacket(books=[Book(id="b1", status="identified", title="Dune", frame_ref="frames/f1.jpg",
                                replacement_cost=BookPrice(amount=20, source="memory"))])
    out = finalize(p, root, None)
    assert out.books[0].replacement_cost.amount is None
    assert out.totals.books_replacement_cost == 0 and out.totals.excluded_from_totals == 1
    assert any("no source" in r.reason for r in out.review_queue)


def test_appraisal_lines_carry_no_price(root):
    p = ClaimPacket(books=[Book(id="b1", status="needs_appraisal", title="Dune", frame_ref="frames/f1.jpg",
                                replacement_cost=BookPrice(amount=900, **GOOD))],
                    items=[Item(id="i1", status="needs_appraisal", frame_ref="frames/f1.jpg",
                                replacement_cost=ItemPrice(low=1, high=2, **GOOD))])
    out = finalize(p, root, None)
    assert out.books[0].replacement_cost.amount is None and out.items[0].replacement_cost.low is None
    assert out.totals.excluded_from_totals == 2


def test_unidentified_cannot_keep_a_title(root):
    p = ClaimPacket(books=[Book(id="b1", status="unidentified", title="Probably Dune", isbn="123", frame_ref="frames/f1.jpg")])
    b = finalize(p, root, None).books[0]
    assert b.title == "" and b.isbn == ""


def test_identified_without_title_downgraded(root):
    p = ClaimPacket(books=[Book(id="b1", status="identified", frame_ref="frames/f1.jpg")])
    assert finalize(p, root, None).books[0].status == "unidentified"


def test_missing_frames_go_to_review(root):
    p = ClaimPacket(books=[Book(id="b1", frame_ref="frames/nope.jpg#xyxy=1,2,3,4"), Book(id="b2")])
    reasons = {r.ref_id: r.reason for r in finalize(p, root, None).review_queue}
    assert "not found" in reasons["b1"] and reasons["b2"] == "no frame reference"


def test_frame_fragment_is_ignored_when_checking(root):
    p = ClaimPacket(books=[Book(id="b1", frame_ref="frames/f1.jpg#xyxy=1,2,3,4")])
    assert not any("frame" in r.reason for r in finalize(p, root, None).review_queue)


def test_negative_price_removed(root):
    p = ClaimPacket(books=[Book(id="b1", status="identified", title="X", frame_ref="frames/f1.jpg",
                                replacement_cost=BookPrice(amount=-5, **GOOD))])
    assert finalize(p, root, None).books[0].replacement_cost.amount is None


def test_item_range_swapped_and_priced_without_low_becomes_range(root):
    p = ClaimPacket(items=[Item(id="i1", status="range", frame_ref="frames/f1.jpg", replacement_cost=ItemPrice(low=90, high=40, **GOOD)),
                           Item(id="i2", status="priced", frame_ref="frames/f1.jpg")])
    out = finalize(p, root, None)
    assert (out.items[0].replacement_cost.low, out.items[0].replacement_cost.high) == (40, 90)
    assert out.items[1].status == "range"


def test_duplicate_ids_rejected(root):
    with pytest.raises(ValueError):
        finalize(ClaimPacket(books=[Book(id="x")], items=[Item(id="x")]), root, None)


def test_review_entries_deduplicated(root):
    p = ClaimPacket(books=[Book(id="b1")])
    p.review_queue = list(finalize(p, root, None).review_queue)
    assert len(finalize(p, root, None).review_queue) == 1


def test_report_escapes_untrusted_text(root):
    p = ClaimPacket(books=[Book(id="b1", status="identified", title="<script>alert(1)</script>", frame_ref="frames/f1.jpg",
                                replacement_cost=BookPrice(amount=10, **GOOD))])
    html = render(finalize(p, root, 1.0))
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "source</a>" in html  # the price macro still renders links


def test_report_renders_empty_packet():
    assert "Contents claim" in render(ClaimPacket())
