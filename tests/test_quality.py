import cv2
import numpy as np
import pytest

from libraryclaim.quality import assess, coaching
from helpers import textured


def test_sharp_frame_passes():
    q, _ = assess(textured())
    assert q.ok, q


def test_blur_is_caught():
    q, _ = assess(cv2.GaussianBlur(textured(), (31, 31), 12))
    assert "blurry" in q.problems


def test_thresholds_independent_of_resolution():
    small = textured(480, 640)
    big = cv2.resize(small, (2560, 1920), interpolation=cv2.INTER_NEAREST)
    assert assess(small)[0].ok and assess(big)[0].ok


def test_glare_patch():
    img = textured()
    img[100:250, 200:400] = 255
    assert "glare" in assess(img)[0].problems


def test_dark_frame_reports_dark_not_blur():
    q, _ = assess((textured() * 0.1).astype(np.uint8))
    assert q.problems[0] == "too_dark"
    assert "blurry" not in q.problems


def test_motion_needs_previous_frame():
    q, _ = assess(textured())
    assert q.motion is None


def test_fast_pan_flagged_and_slow_pan_not():
    base = textured(480, 900)
    a, b_fast, b_slow = base[:, :640], base[:, 60:700], base[:, 3:643]
    _, prev = assess(a)
    assert "moving_too_fast" in assess(b_fast, prev)[0].problems
    assert "moving_too_fast" not in assess(b_slow, prev)[0].problems


def test_grayscale_and_odd_sizes_accepted():
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    assert assess(gray)[0].ok
    assess(textured(17, 33))  # tiny frame must not crash


def test_prev_frame_of_different_size():
    _, prev = assess(textured(480, 640))
    q, _ = assess(textured(720, 1280))
    assert q is not None
    assess(textured(720, 1280), prev)  # resized, no crash


def test_empty_frame_rejected():
    with pytest.raises(ValueError):
        assess(np.zeros((0, 0, 3), np.uint8))


def test_coaching_prioritises_motion():
    assert "slow down" in coaching(["glare", "moving_too_fast"])
    assert coaching([]) == ""
