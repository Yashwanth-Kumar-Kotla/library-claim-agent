import json

import cv2
import numpy as np
import pytest

from libraryclaim.config import Locale, Settings
from libraryclaim.live.session import LiveSweep, keyframes
from libraryclaim.pipeline import Deps
from helpers import FakeWeb, textured
from synthetic import ROWS, FakeVision, draw_row, fake_catalogue

SETTINGS = Settings(locale=Locale("US", "USD"))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def jpeg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


@pytest.fixture
def live(tmp_path):
    clock = Clock()
    events = []
    s = LiveSweep(tmp_path / "sw", Deps(vision=FakeVision(), get_json=FakeWeb(fake_catalogue()), ebay_token="t"),
                  SETTINGS, on_event=lambda k, d: events.append(k), clock=clock)
    return s, clock, events


def pan(s, clock, img, steps, step_px, dt=0.33):
    """Feed frames that pan across img left to right."""
    h, w = img.shape[:2]
    view = min(w, 640)
    out = []
    for i in range(steps):
        x = min(i * step_px, w - view)
        clock.t += dt
        out.append(s.add_frame(jpeg(img[:, x:x + view])))
    return out


def spoken(results):
    return [r[0] for r in results if r[0]]


def screen(results):
    return [r[1] for r in results if r[1]]


def blur_frames(s, clock, n, dt=0.33):
    blurry = jpeg(cv2.GaussianBlur(textured(), (31, 31), 12))
    out = []
    for _ in range(n):
        clock.t += dt
        out.append(s.add_frame(blurry))
    return out


def test_locale_set_once_before_shelves(live):
    s, _, _ = live
    assert s.set_locale("gb", "gbp")["currency"] == "GBP"
    assert s.set_locale("Britain", "GBP")["ok"] is False
    s.start_row("A1")
    s.add_frame(jpeg(draw_row(ROWS["A1"])[0]))
    s.add_frame(jpeg(draw_row(ROWS["A1"])[0]))
    s.finish_row()
    assert s.set_locale("US", "USD")["ok"] is False


def test_row_names_validated(live):
    s, _, _ = live
    assert not s.start_row("shelf one")["ok"] and s.start_row("a2")["row"] == "A2"


def test_every_frame_saved_and_bad_bytes_ignored(live):
    s, _, _ = live
    s.add_frame(jpeg(textured()))
    assert s.add_frame(b"not a jpeg") == (None, None)
    assert len(list((s.root / "frames").iterdir())) == 1


def test_row_with_too_few_clear_frames_is_rejected_with_advice(live):
    s, clock, events = live
    s.start_row("A1")
    s.add_frame(jpeg(cv2.GaussianBlur(textured(), (31, 31), 12)))
    r = s.finish_row()
    assert not r["ok"] and "go back over A1" in r["advice"] and "row_rejected" in events
    assert s.rows_done == []


def test_good_row_goes_to_pipeline_and_frames_outside_rows_are_room_views(live):
    s, clock, events = live
    for _ in range(10):  # room views before the shelves: windows of 2.5 s
        clock.t += 0.33
        s.add_frame(jpeg(textured(seed=5)))
    s.start_row("A1")
    img = draw_row(ROWS["A1"])[0]
    for _ in range(3):
        clock.t += 0.33
        s.add_frame(jpeg(img))
    r = s.finish_row()
    assert r["ok"] and r["rows_done"] == ["A1"]
    assert len(s.room_frames) >= 1
    out = s.end()
    assert out["ok"] and "books counted" in out["summary"]
    packet = json.loads((s.root / "claim_packet.json").read_text())
    assert len([b for b in packet["books"] if b["shelf"] == "A1"]) == len(ROWS["A1"])


def test_starting_a_row_closes_the_open_one(live):
    s, clock, _ = live
    img = draw_row(ROWS["A1"])[0]
    s.start_row("A1")
    for _ in range(3):
        s.add_frame(jpeg(img))
    r = s.start_row("A2")
    assert r["previous_row"]["row"] == "A1" and r["previous_row"]["ok"]


def test_missing_rows_block_the_end_until_confirmed(live):
    s, clock, _ = live
    img = draw_row(ROWS["A1"])[0]
    for shelf in ("A1", "A3"):
        s.start_row(shelf)
        for _ in range(3):
            s.add_frame(jpeg(img))
        s.finish_row()
    first = s.end()
    assert not first["ok"] and first["missing_rows"] == ["A2"]
    assert s.end(confirmed=True)["ok"]


def test_frames_after_the_end_are_ignored(live):
    s, _, _ = live
    s.end()
    assert s.add_frame(jpeg(textured())) == (None, None)


def test_remarks_need_enough_to_be_tied_to_a_line(live):
    s, _, _ = live
    assert not s.record_remark("first_edition", "that's a first edition")["ok"]  # which book?
    assert not s.record_remark("skip_shelf", "skip that")["ok"]                 # which shelf?
    s.start_row("B2")
    assert s.record_remark("skip_shelf", "skip that one, not mine")["applied_to"] == "B2"
    assert s.record_remark("signed", "Dune is signed", title="Dune")["ok"]
    assert not s.record_remark("haunted", "it's haunted")["ok"]


def test_keyframes_span_the_pan_and_keep_the_sharpest():
    from libraryclaim.live.session import Frame
    from libraryclaim.quality import FrameQuality
    frames = [Frame(f"f{i}", i, FrameQuality(100 + i % 3, 0, 120, 64.0 if i else None), 100 + i % 3) for i in range(20)]
    keys = keyframes(frames)
    assert keys[0].ref == "f0" and keys[-1].ref in ("f19", "f18", "f17")
    assert 5 <= len(keys) <= 8  # 20 frames x 10% of a frame width = 2 widths of travel -> ~6 keyframes
    assert keyframes(frames[:2]) == frames[:2]


def test_still_camera_sends_one_room_view_not_dozens(live):
    s, clock, _ = live
    still = jpeg(textured(seed=9))
    for _ in range(80):  # ~26 s of a phone or webcam that doesn't move
        clock.t += 0.33
        s.add_frame(still)
    s.start_row("A1")  # flushes the open window
    assert len(s.room_frames) == 1


def test_panning_camera_sends_a_view_per_new_stretch(live):
    s, clock, _ = live
    pan(s, clock, textured(480, 6000, seed=4), steps=60, step_px=90)
    assert 3 <= len(s.room_frames) <= 12


def test_end_after_disconnect_builds_a_packet_and_says_so(live):
    s, clock, _ = live
    s.add_frame(jpeg(textured()))
    out = s.end(True, closed_early=True)
    packet = json.loads((s.root / "claim_packet.json").read_text())
    assert out["ok"] and any("connection closed" in r["reason"] for r in packet["review_queue"])
    assert s.end()["ok"] is False  # never twice




def test_short_blur_shows_on_screen_but_is_not_spoken(live):
    s, clock, _ = live
    clock.t = 20
    s.start_row("A1")
    r = blur_frames(s, clock, 6)  # 2 seconds
    assert screen(r) and not spoken(r)


def test_lasting_blur_on_a_shelf_is_spoken_once_per_shelf(live):
    s, clock, _ = live
    clock.t = 20
    s.start_row("A1")
    r = blur_frames(s, clock, 60)  # 20 seconds of blur
    assert len(spoken(r)) == 1 and "blurry" in spoken(r)[0]
    s.start_row("A2")
    clock.t += 25
    assert len(spoken(blur_frames(s, clock, 30))) == 1  # new shelf, may say it once more


def test_nothing_spoken_in_the_opening_seconds(live):
    s, clock, _ = live
    s.start_row("A1")
    assert not spoken(blur_frames(s, clock, 25))  # first ~8 s


def test_no_spine_warnings_while_showing_the_room(live):
    s, clock, _ = live
    clock.t = 20
    r = pan(s, clock, textured(480, 4000), steps=30, step_px=300) + blur_frames(s, clock, 30)
    assert not spoken(r) and not screen(r)
    dark = jpeg((textured() * 0.1).astype("uint8"))
    r = []
    for _ in range(30):
        clock.t += 0.33
        r.append(s.add_frame(dark))
    assert spoken(r) and "dark" in spoken(r)[0]  # darkness matters anywhere


def test_real_handheld_motion_is_not_called_too_fast(live):
    s, clock, _ = live
    clock.t = 20
    s.start_row("A1")
    wide = np.hstack([textured(480, 640, seed=i) for i in range(4)])  # dense texture all the way along
    r = pan(s, clock, wide, steps=40, step_px=20)  # ~median motion of a real phone sweep
    assert not spoken(r) and not screen(r)


def test_nudge_when_camera_moves_with_no_shelf_open(live):
    s, clock, _ = live
    img = draw_row(ROWS["A1"])[0]
    assert s.nudge() is None  # nothing before the first shelf
    s.start_row("A1")
    for _ in range(3):
        s.add_frame(jpeg(img))
    s.finish_row()
    clock.t += 3
    pan(s, clock, textured(480, 4000, seed=3), steps=4, step_px=200)
    assert s.nudge() is None  # they are probably walking to the next bookcase
    clock.t += 30
    assert "start_row" in (s.nudge() or "")
    assert s.nudge() is None  # not repeated straight away
    clock.t += 46
    assert s.nudge()
    s.start_row("A2")
    clock.t += 60
    assert s.nudge() is None  # a shelf is open


def turn(s, clock, headings, beta=90.0):
    img = jpeg(textured(seed=11))
    for a in headings:
        clock.t += 0.33
        s.add_frame(img, (float(a), beta, 0.0))


def test_full_turn_with_tilt_is_accepted_and_saved(live):
    s, clock, _ = live
    s.start_room_scan()
    turn(s, clock, range(0, 360, 10))
    turn(s, clock, [0], beta=125)
    turn(s, clock, [0], beta=55)
    r = s.finish_room_scan()
    assert r["ok"] and r["covered_deg"] >= 300
    saved = json.loads((s.root / "scan.json").read_text())
    assert len(saved) == 38 and {"ref", "alpha", "beta", "gamma"} <= set(saved[0])


def test_half_turn_or_no_tilt_asks_for_a_redo(live):
    s, clock, _ = live
    s.start_room_scan()
    turn(s, clock, range(0, 180, 10))
    r = s.finish_room_scan()
    assert not r["ok"] and "all the way round" in r["advice"]
    s.start_room_scan()
    turn(s, clock, range(0, 360, 10))
    r = s.finish_room_scan()
    assert not r["ok"] and "tilt" in r["advice"] and not (s.root / "scan.json").exists()


def test_phone_without_orientation_is_told_plainly(live):
    s, clock, _ = live
    s.start_room_scan()
    for _ in range(10):
        s.add_frame(jpeg(textured()))
    r = s.finish_room_scan()
    assert not r["ok"] and "did not share its orientation" in r["advice"]


def test_scan_closes_an_open_shelf_and_frames_still_count_as_room_views(live):
    s, clock, _ = live
    img = draw_row(ROWS["A1"])[0]
    s.start_row("A1")
    for _ in range(3):
        s.add_frame(jpeg(img))
    r = s.start_room_scan()
    assert r["previous_row"]["row"] == "A1"
    clock.t = 30
    turn(s, clock, range(0, 360, 10))
    assert s.room_frames  # the turn also feeds item detection
    assert s.finish_room_scan()["ok"] is False  # no tilt
    assert s.finish_room_scan()["ok"] is False and "no room scan is open" in s.finish_room_scan()["error"]


def test_a_live_session_replays_to_the_same_packet(live):
    from libraryclaim.pipeline import Deps, run_pipeline
    s, clock, _ = live
    clock.t = 20
    for shelf in ("A1", "A2"):
        s.start_row(shelf)
        for _ in range(3):
            clock.t += 0.33
            s.add_frame(jpeg(draw_row(ROWS[shelf])[0]))
        s.finish_row()
    s.start_row("A1")                                   # re-capture
    for _ in range(3):
        clock.t += 0.33
        s.add_frame(jpeg(draw_row(ROWS["A1"])[0]))
    s.finish_row()
    s.record_remark("signed", "the Dune is signed", title="Dune", shelf="A1")
    s.start_row("B1")
    s.record_remark("skip_shelf", "skip it, not mine")
    for _ in range(10):
        clock.t += 0.33
        s.add_frame(jpeg(textured(seed=7)))
    live_packet = s.end(confirmed=True)
    assert live_packet["ok"]
    first = json.loads((s.root / "claim_packet.json").read_text())
    assert (s.root / "out" / "claim_packet.live.json").is_file()

    replayed = run_pipeline(s.root, Deps(vision=FakeVision(), get_json=FakeWeb(fake_catalogue()), ebay_token="t"), SETTINGS)
    again = replayed.model_dump(mode="json")
    strip = lambda p: {k: v for k, v in p.items() if k != "sweep"}
    assert strip(again) == strip(first)
    dune = next(b for b in again["books"] if b["title"] == "Dune")
    assert dune["status"] == "needs_appraisal" and dune["id"].endswith("t1")   # remark and re-capture both replayed
    assert json.loads((s.root / "out" / "claim_packet.live.json").read_text()) == first
