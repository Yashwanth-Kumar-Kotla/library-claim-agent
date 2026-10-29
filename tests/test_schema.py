from libraryclaim.schema import Book, BookPrice, ClaimPacket, Item, ItemPrice, compute_totals


def test_totals_sum_lines_and_exclude_unpriced_and_appraisal():
    packet = ClaimPacket(
        books=[
            Book(id="b1", status="identified", replacement_cost=BookPrice(amount=20), used_value=BookPrice(amount=8)),
            Book(id="b2", status="identified", replacement_cost=BookPrice(amount=15.5)),
            Book(id="b3", status="unidentified"),
            Book(id="b4", status="needs_appraisal", replacement_cost=BookPrice(amount=900)),
        ],
        items=[
            Item(id="i1", status="priced", replacement_cost=ItemPrice(low=100, high=100)),
            Item(id="i2", status="range", replacement_cost=ItemPrice(low=40, high=90)),
            Item(id="i3", status="needs_appraisal"),
        ],
    )
    t = compute_totals(packet, shelf_run_m=3.2)
    assert t.book_count == 4
    assert t.books_identified == 2
    assert t.books_unidentified == 1
    assert t.books_replacement_cost == 35.5
    assert t.books_used_value == 8
    assert t.items_replacement_cost_low == 140
    assert t.items_replacement_cost_high == 190
    assert t.excluded_from_totals == 3
    assert t.shelf_run_m == 3.2


def test_empty_fields_serialize_as_unknown_not_zero():
    data = ClaimPacket().model_dump()
    assert data["room"]["floor_area_m2"] is None
    assert {"sweep", "room", "books", "items", "totals", "review_queue"} <= set(data)  # added fields allowed, none dropped
