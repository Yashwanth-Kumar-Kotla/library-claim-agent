import pytest

from libraryclaim.boxes import Box, extract_json, merge_tiles, parse_boxes, tiles


def test_fenced_json_with_chatter():
    reply = 'Here you go:\n```json\n[{"box_2d": [100, 200, 900, 260], "label": "spine"}]\n```\nHope that helps'
    boxes, dropped = parse_boxes(reply, 1000, 1000)
    assert len(boxes) == 1 and dropped == 0
    b = boxes[0]
    assert (b.x0, b.y0, b.x1, b.y1) == (200, 100, 260, 900)


def test_object_wrapper_and_pixel_conversion_with_offset():
    reply = '{"books": [{"box_2d": [0, 0, 500, 100]}]}'
    (b,), _ = parse_boxes(reply, width=400, height=800, offset=(1000, 50))
    assert (b.x0, b.y0, b.x1, b.y1) == (1000, 50, 1040, 450)


def test_inverted_and_out_of_range_boxes_repaired():
    (b,), _ = parse_boxes('[{"box_2d": [900, 260, 100, -50]}]', 1000, 1000)
    assert (b.x0, b.x1, b.y0, b.y1) == (0, 260, 100, 900)


@pytest.mark.parametrize("item", [
    '{"label": "no box"}',
    '{"box_2d": [1, 2, 3]}',
    '{"box_2d": ["a", "b", "c", "d"]}',
    '{"box_2d": null}',
    '{"box_2d": [100, 100, 100, 500]}',  # zero height
    '"just a string"',
])
def test_garbage_items_dropped_and_counted(item):
    boxes, dropped = parse_boxes(f'[{item}, {{"box_2d": [0, 0, 900, 50]}}]', 1000, 1000)
    assert len(boxes) == 1 and dropped == 1


@pytest.mark.parametrize("reply", ["", "I can't see any books.", "[]", "{}", "null", "[{]"])
def test_unusable_replies_give_no_boxes(reply):
    assert parse_boxes(reply, 100, 100) == ([], 0)


def test_none_reply():
    with pytest.raises(ValueError):
        extract_json(None)


def test_lying_flat_flag_parsing():
    reply = '[{"box_2d":[0,0,50,900],"lying_flat":"true"},{"box_2d":[0,0,900,50],"lying_flat":false},{"box_2d":[0,0,900,60],"lying_flat":"maybe"}]'
    boxes, _ = parse_boxes(reply, 1000, 1000)
    assert [b.lying_flat for b in boxes] == [True, False, None]


def test_extra_fields_kept():
    (b,), _ = parse_boxes('[{"box_2d":[0,0,900,50],"text":"DUNE"}]', 1000, 1000)
    assert b.extra == {"text": "DUNE"}


def test_tiles_cover_whole_width_with_overlap():
    ts = tiles(2500, 600, 1000, 0.25)
    assert ts[0][0] == 0 and ts[-1][2] == 2500
    for (a, _, a1, _), (b, _, _, _) in zip(ts, ts[1:]):
        assert b < a1  # every neighbour overlaps


def test_narrow_image_is_one_tile():
    assert tiles(800, 600, 1000) == [(0, 0, 800, 600)]
    assert tiles(1000, 600, 1000) == [(0, 0, 1000, 600)]


def test_spine_cut_at_tile_edge_counted_once():
    whole = Box(980, 10, 1010, 400)
    clipped = Box(980, 10, 1000, 400)   # same spine, cut by the first tile's right edge
    assert len(merge_tiles([clipped, whole])) == 1
    assert merge_tiles([clipped, whole])[0].x1 == 1010


def test_adjacent_thin_spines_not_merged():
    row = [Box(x, 10, x + 12, 400) for x in range(0, 120, 12)]  # 10 touching 12px spines
    assert len(merge_tiles(row + row[3:6])) == 10


def test_flat_stack_books_kept_separate():
    stack = [Box(100, 300 + i * 20, 400, 320 + i * 20) for i in range(5)]
    assert len(merge_tiles(stack)) == 5


def test_merge_output_is_left_to_right():
    out = merge_tiles([Box(50, 0, 60, 10), Box(0, 0, 10, 10), Box(20, 0, 30, 10)])
    assert [b.x0 for b in out] == [0, 20, 50]


def test_merge_keeps_the_more_legible_read():
    clipped = Box(980, 10, 1000, 400, extra={"text": "DU", "legibility": 0.2})
    whole = Box(975, 10, 1010, 400, extra={"text": "DUNE FRANK HERBERT", "legibility": 0.9})
    weak_whole = Box(975, 10, 1010, 400, extra={"text": "D", "legibility": 0.1})
    strong_clip = Box(980, 10, 1000, 400, extra={"text": "DUNE FRANK HERBERT", "legibility": 0.9})
    assert merge_tiles([clipped, whole])[0].extra["text"] == "DUNE FRANK HERBERT"
    out = merge_tiles([weak_whole, strong_clip])
    assert len(out) == 1 and out[0].x0 == 975 and out[0].extra["text"] == "DUNE FRANK HERBERT"


def test_real_leaning_spine_seen_twice_with_same_text_is_one_book():
    a = Box(100, 0, 160, 400, extra={"text": "LAINI TAYLOR STRANGE THE DREAMER"})
    b = Box(130, 0, 190, 400, extra={"text": "Laini Taylor  Strange the Dreamer"})  # 50% overlap
    assert len(merge_tiles([a, b])) == 1


def test_overlapping_neighbours_with_different_text_stay_apart():
    a = Box(100, 0, 160, 400, extra={"text": "STRANGE THE DREAMER"})
    b = Box(130, 0, 190, 400, extra={"text": "MUSE OF NIGHTMARES"})
    assert len(merge_tiles([a, b])) == 2


def test_real_box_key_aliases_accepted():
    # Seen live: with low thinking the items reply used "box" instead of "box_2d", dropping every item.
    reply = '[{"box":[0,0,500,500],"category":"decor"},{"bbox":[0,0,400,400]},{"bounding_box":[0,0,300,300]}]'
    boxes, dropped = parse_boxes(reply, 1000, 1000)
    assert len(boxes) == 3 and dropped == 0
    assert "box" not in boxes[0].extra and boxes[0].extra == {"category": "decor"}


def test_real_leaning_books_in_one_tile_are_never_merged_on_geometry():
    # Bottom-shelf books lean, so their boxes overlap neighbours by more than half.
    leaning = [Box(100 + i * 25, 0, 160 + i * 25, 400, tile=0) for i in range(8)]
    assert len(merge_tiles(leaning)) == 8


def test_cross_tile_duplicate_still_merged():
    a = Box(980, 10, 1000, 400, tile=0, extra={"text": "DUNE FR"})
    b = Box(975, 10, 1010, 400, tile=1, extra={"text": "DUNE FRANK HERBERT"})
    assert len(merge_tiles([a, b])) == 1


def test_cross_tile_overlap_with_different_books_kept():
    a = Box(100, 0, 160, 400, tile=0, extra={"text": "STRANGE THE DREAMER"})
    b = Box(105, 0, 165, 400, tile=1, extra={"text": "MUSE OF NIGHTMARES"})
    assert len(merge_tiles([a, b])) == 2


def test_same_tile_listed_twice_with_same_text_merged():
    a = Box(100, 0, 160, 400, tile=0, extra={"text": "THE JOY LUCK CLUB AMY TAN"})
    b = Box(110, 0, 165, 400, tile=0, extra={"text": "The Joy Luck Club  Amy Tan"})
    assert len(merge_tiles([a, b])) == 1


def test_real_broken_entry_does_not_lose_the_tile():
    # Verbatim shape of a live reply: one entry has a key with no value.
    reply = '''[
  {"box_2d": [267, 0, 919, 46], "lying_flat": false, "text": "DINI TAYLOR", "legibility": 0.5},
  {"box_2d": [98, 755, 634, 856], "label": "lying_flat": false, "text": "HOUSE OF NAMES", "legibility": 0.8},
  {"box_2d": [334, 75, 897, 151], "lying_flat": false, "text": "DOGS AT THE PERIMETER", "legibility": 0.7}
]'''
    boxes, dropped = parse_boxes(reply, 1000, 1000)
    assert [b.extra["text"] for b in boxes] == ["DINI TAYLOR", "HOUSE OF NAMES", "DOGS AT THE PERIMETER"]
    assert dropped == 0


def test_unrepairable_entry_dropped_alone_and_counted():
    reply = '[{"box_2d": [0,0,900,50], "text": "OK"}, {"box_2d": [0,0,900 "text": }, {"box_2d": [0,60,900,90], "text": "ALSO OK"}]'
    boxes, dropped = parse_boxes(reply, 1000, 1000)
    assert [b.extra["text"] for b in boxes] == ["OK", "ALSO OK"] and dropped == 1


def test_braces_and_quotes_inside_text_do_not_confuse_recovery():
    reply = '[{"box_2d": [0,0,900,50], "text": "A {weird} \\"title\\""}, {"box_2d": [0,60,900 "x": }]'
    boxes, dropped = parse_boxes(reply, 1000, 1000)
    assert boxes[0].extra["text"] == 'A {weird} "title"' and dropped == 1


def test_real_misspelled_box_key():
    boxes, _ = parse_boxes('[{"box_2": [390, 0, 1000, 69], "text": "THE DAYS"}]', 1000, 1000)
    assert len(boxes) == 1 and "box_2" not in boxes[0].extra


def test_real_duplicates_after_union_are_removed():
    from libraryclaim.boxes import dedupe_reads
    shiver_a = Box(1494, 225, 1549, 638, extra={"title": "Shiver", "text": "", "legibility": 0.5})
    shiver_b = Box(1526, 241, 1551, 638, extra={"title": "shiver", "text": "stiefvater shiver", "legibility": 0.9})
    lament = Box(1474, 235, 1518, 638, extra={"title": "LAMENT", "text": "LAMENT S TER"})
    days_a = Box(337, 182, 575, 750, extra={"title": "Days of BLOOD & STARLIGHT"})
    days_b = Box(436, 155, 600, 741, extra={"title": "Days of BLOOD & STARLIGHT"})
    out = dedupe_reads([lament, shiver_a, shiver_b, days_a, days_b])
    assert len(out) == 3
    assert next(b for b in out if title_has(b, "shiver")).extra["text"] == "stiefvater shiver"


def title_has(b, word):
    return word in b.extra.get("title", "").lower()


def test_two_real_copies_side_by_side_are_kept():
    from libraryclaim.boxes import dedupe_reads
    a = Box(100, 0, 140, 400, extra={"title": "The Hobbit"})
    b = Box(141, 0, 181, 400, extra={"title": "The Hobbit"})
    assert len(dedupe_reads([a, b])) == 2


def test_overlapping_leaning_books_with_different_titles_kept():
    from libraryclaim.boxes import dedupe_reads
    a = Box(100, 0, 200, 400, extra={"title": "Strange the Dreamer"})
    b = Box(150, 0, 250, 400, extra={"title": "Muse of Nightmares"})
    c = Box(160, 0, 260, 400, extra={})  # unread spine overlapping both: never merged on no evidence
    assert len(dedupe_reads([a, b, c])) == 3


def test_packed_spans_narrow_loose_boxes_to_neighbour_spacing():
    from libraryclaim.boxes import Box, packed_spans
    # three upright spines 20 px apart, boxed 60 px wide; a flat book keeps its own width
    boxes = [Box(80, 0, 140, 300), Box(100, 0, 160, 300), Box(120, 0, 180, 300), Box(300, 250, 500, 290, lying_flat=True)]
    spans = packed_spans(boxes)
    assert round(spans[1][2] - spans[1][0]) == 20                       # the middle one, overlapped on both sides
    assert spans[0][0] == 80 and spans[2][2] == 180                     # outer edges are not touched
    assert spans[3] == boxes[3].as_tuple()
    assert packed_spans([Box(0, 0, 30, 300)]) == [(0, 0, 30, 300)]   # alone on the shelf: unchanged
    thick_thin = [Box(0, 0, 50, 300), Box(50, 0, 60, 300)]             # touching, not overlapping
    assert packed_spans(thick_thin) == [b.as_tuple() for b in thick_thin]
