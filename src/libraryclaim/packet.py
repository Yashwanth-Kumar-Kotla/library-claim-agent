"""Final checks before a packet leaves the system.

Rules from the brief, enforced in code rather than trusted to earlier stages:
- a filled field with no evidence is a defect: a price without source, URL and date is removed;
- every frame_ref must open: a missing frame sends the line to review;
- nothing awaiting appraisal carries a price;
- totals are summed from the lines here.
"""

from __future__ import annotations

from pathlib import Path

from .schema import Book, ClaimPacket, Item, ReviewEntry, compute_totals


def frame_path(ref: str) -> str:
    return ref.split("#", 1)[0]


def has_evidence(source: str, url: str, retrieved_at: str) -> bool:
    return bool(source.strip() and url.strip() and retrieved_at.strip())


def enforce_book(b: Book, root: Path, review: list[ReviewEntry]) -> None:
    for name in ("replacement_cost", "used_value"):
        p = getattr(b, name)
        if p.amount is not None and not has_evidence(p.source, p.url, p.retrieved_at):
            review.append(ReviewEntry(ref_id=b.id, reason=f"{name} removed: no source, URL or retrieval date"))
            setattr(b, name, type(p)())
        elif p.amount is not None and p.amount < 0:
            review.append(ReviewEntry(ref_id=b.id, reason=f"{name} removed: negative amount"))
            setattr(b, name, type(p)())
    if b.status == "needs_appraisal":
        b.replacement_cost = type(b.replacement_cost)()
        b.used_value = type(b.used_value)()
    if b.status == "identified" and not b.title:
        b.status = "unidentified"
        review.append(ReviewEntry(ref_id=b.id, reason="marked identified without a title; downgraded"))
    if b.status == "unidentified":
        # Unidentified means we don't know what it is: no title, author or ISBN may survive.
        b.title = b.author = b.edition = b.isbn = ""
    check_frame(b.id, b.frame_ref, root, review)


def enforce_item(it: Item, root: Path, review: list[ReviewEntry]) -> None:
    p = it.replacement_cost
    if (p.low is not None or p.high is not None) and not has_evidence(p.source, p.url, p.retrieved_at):
        review.append(ReviewEntry(ref_id=it.id, reason="price removed: no source, URL or retrieval date"))
        it.replacement_cost = type(p)()
    if it.status == "needs_appraisal":
        it.replacement_cost = type(p)()
    p = it.replacement_cost
    if p.low is not None and p.high is not None and p.low > p.high:
        p.low, p.high = p.high, p.low
    if it.status == "priced" and p.low is None:
        it.status = "range"
    check_frame(it.id, it.frame_ref, root, review)


def check_frame(line_id: str, ref: str, root: Path, review: list[ReviewEntry]) -> None:
    if not ref:
        review.append(ReviewEntry(ref_id=line_id, reason="no frame reference"))
    elif not (root / frame_path(ref)).is_file():
        review.append(ReviewEntry(ref_id=line_id, reason=f"frame {frame_path(ref)} not found"))


def finalize(packet: ClaimPacket, root: Path, shelf_run_m: float | None) -> ClaimPacket:
    review = list(packet.review_queue)
    ids = [b.id for b in packet.books] + [i.id for i in packet.items]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate line ids: {sorted(dupes)}")
    for b in packet.books:
        enforce_book(b, root, review)
    for it in packet.items:
        enforce_item(it, root, review)
    seen, unique = set(), []
    for r in review:
        if (r.ref_id, r.reason) not in seen:
            seen.add((r.ref_id, r.reason))
            unique.append(r)
    packet.review_queue = unique
    packet.totals = compute_totals(packet, shelf_run_m)
    return packet
