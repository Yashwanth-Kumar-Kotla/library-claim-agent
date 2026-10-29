import json

import numpy as np
import pytest

from libraryclaim.boxes import Box
from libraryclaim.vision import (detect_items, detect_spines, parse_read, read_spine, spine_crop)


class ScriptedModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, jpeg, prompt):
        self.calls += 1
        assert jpeg[:2] == b"\xff\xd8"  # always a real JPEG
        r = self.replies.pop(0) if self.replies else "[]"
        return r(prompt) if callable(r) else r


def test_read_ok():
    t = parse_read('{"text":"DUNE FRANK HERBERT ACE","title":"Dune","author":"Frank Herbert","publisher":"Ace","legibility":0.9}')
    assert t.title == "Dune" and t.readable


@pytest.mark.parametrize("reply", ["", "nope", "[]", "null", '{"text":"","legibility":0.9}',
                                   '{"title":"Unknown","author":"N/A","text":"illegible","legibility":0.8}'])
def test_unreadable_reads_have_zero_legibility(reply):
    t = parse_read(reply)
    assert not t.readable and t.legibility == 0


def test_legibility_percent_and_garbage_normalised():
    assert parse_read('{"text":"DUNE","legibility":85}').legibility == 0.85
    assert parse_read('{"text":"DUNE","legibility":"high"}').legibility == 0
    assert parse_read('{"text":"DUNE","legibility":-2}').legibility == 0


def test_short_fragment_not_readable():
    assert not parse_read('{"text":"THE","legibility":0.9}').readable


def test_whitespace_collapsed():
    assert parse_read('{"text":"  THE\\n  HOBBIT ","legibility":1}').text == "THE HOBBIT"


class StripeModel:
    """Finds bright vertical stripes in whatever tile it is shown; thread-safe call counter."""

    def __init__(self, drop_every_other=False):
        import threading
        self.calls, self.lock, self.drop = 0, threading.Lock(), drop_every_other

    def __call__(self, jpeg, prompt):
        import cv2
        with self.lock:
            self.calls += 1
            n = self.calls
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        cols = np.where(img.mean(axis=0) > 128)[0]
        runs = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1) if cols.size else []
        out = [{"box_2d": [50, r[0] * 1000 / w, 950, (r[-1] + 1) * 1000 / w], "text": f"S{r[0]}"} for r in runs if r.size]
        if self.drop and n % 2:
            out = out[::2]  # an unreliable pass that misses every other spine
        return json.dumps(out)


def test_detect_spines_tiles_passes_and_dedupes():
    img = np.zeros((600, 2400, 3), np.uint8)
    img[:, 1040:1080] = 255  # one spine inside the overlap of both tiles
    model = StripeModel()
    boxes, dropped = detect_spines(model, img, tile_w=1400)
    assert model.calls == 4  # 2 tiles x 2 passes
    assert len(boxes) == 1 and dropped == 0
    assert boxes[0].x0 == pytest.approx(1040, abs=4) and boxes[0].x1 == pytest.approx(1080, abs=4)


def test_second_pass_recovers_spines_the_first_missed():
    img = np.zeros((600, 1000, 3), np.uint8)
    for x in range(50, 950, 60):
        img[:, x:x + 40] = 255  # 15 spines with gaps
    one = detect_spines(StripeModel(drop_every_other=True), img, tile_w=2000, passes=1)[0]
    two = detect_spines(StripeModel(drop_every_other=True), img, tile_w=2000, passes=2)[0]  # one tile: one bad pass, one good
    assert len(one) < 15 and len(two) == 15


def test_spine_crop_rotates_upright_spines_only():
    img = np.zeros((500, 500, 3), np.uint8)
    up = spine_crop(img, Box(100, 50, 130, 450))
    flat = spine_crop(img, Box(50, 100, 450, 130))
    assert up.shape[1] > up.shape[0]
    assert flat.shape[1] > flat.shape[0]


def test_spine_crop_outside_image_rejected():
    with pytest.raises(ValueError):
        spine_crop(np.zeros((100, 100, 3), np.uint8), Box(200, 200, 220, 260))


def test_detection_read_used_and_escalates_only_when_unsure():
    img = np.full((400, 400, 3), 128, np.uint8)
    sure = Box(100, 20, 130, 380, extra={"text": "DUNE", "title": "Dune", "legibility": 0.9})
    unused = ScriptedModel([])
    assert read_spine(img, sure, escalate=unused).title == "Dune" and unused.calls == 0

    weak = Box(100, 20, 130, 380, extra={"text": "D NE", "legibility": 0.3})
    strong = ScriptedModel(['{"text":"DUNE","title":"Dune","legibility":0.8}'])
    assert read_spine(img, weak, escalate=strong).title == "Dune" and strong.calls == 1

    worse = ScriptedModel(['{"text":"","legibility":0}'])
    assert read_spine(img, weak, escalate=worse).text == "D NE"
    assert read_spine(img, weak, escalate=None).text == "D NE"


def test_detection_without_text_fields_is_unreadable_not_an_error():
    img = np.full((400, 400, 3), 128, np.uint8)
    assert not read_spine(img, Box(100, 20, 130, 380)).readable


def test_items_categories_normalised_and_books_skipped():
    reply = json.dumps([
        {"box_2d": [0, 0, 500, 500], "category": "Lighting", "description": "brass floor lamp"},
        {"box_2d": [0, 0, 500, 500], "category": "spaceship", "description": "odd thing"},
        {"box_2d": [0, 0, 500, 500], "category": "other", "description": "pile of books"},
        {"box_2d": [0, 0, 500, 500], "category": "decor", "description": "framed portrait", "is_art": True},
        {"box_2d": [0, 0, 500, 500], "category": "electronics", "brand_model": "unknown"},
    ])
    items = detect_items(ScriptedModel([reply]), np.zeros((300, 300, 3), np.uint8))
    assert [i.category for i in items] == ["lighting", "other", "art", "electronics"]
    assert items[2].is_art and items[3].brand_model == ""


def test_meter_prices_tokens_and_flags_unpriced_models():
    from libraryclaim.vision import Meter
    m = Meter({"flash": (0.75, 3.75)})
    m.add("flash", 1_000_000, 200_000)
    m.add("mystery", 10, 10)
    c = m.cost()
    assert c["by_model"]["flash"]["usd"] == 0.75 + 0.75
    assert c["by_model"]["mystery"]["priced"] is False and c["usd"] == 1.5


def test_cache_replays_identical_requests_only(tmp_path):
    from libraryclaim.vision import cached
    inner = ScriptedModel(["first", "second"])
    jpeg = b"\xff\xd8 image"
    assert cached(inner, tmp_path, "flash")(jpeg, "p") == "first"
    assert cached(inner, tmp_path, "flash")(jpeg, "p") == "first" and inner.calls == 1  # a later run: from disk
    assert cached(inner, tmp_path, "flash")(jpeg, "other prompt") == "second" and inner.calls == 2
    assert cached(inner, tmp_path, "pro")(jpeg, "p") == "[]"  # other model, own cache entry


def test_cache_keeps_repeated_identical_requests_separate(tmp_path):
    from libraryclaim.vision import cached
    inner = ScriptedModel(["pass one", "pass two"])
    jpeg = b"\xff\xd8 tile"
    first_run = cached(inner, tmp_path, "flash")
    assert [first_run(jpeg, "p"), first_run(jpeg, "p")] == ["pass one", "pass two"]
    replay = cached(ScriptedModel([]), tmp_path, "flash")
    assert [replay(jpeg, "p"), replay(jpeg, "p")] == ["pass one", "pass two"]
