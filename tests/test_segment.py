import numpy as np
import pytest

from libraryclaim.segment import row_image
from helpers import textured


def test_single_frame_used_as_is():
    img = textured()
    r = row_image([("f1", img)])
    assert r.method == "single_frame" and r.image is img


def test_overlapping_frames_stitch_wider_than_any_frame():
    pano = textured(480, 1600, seed=3)
    frames = [(f"f{i}", pano[:, x:x + 640].copy()) for i, x in enumerate((0, 400, 800, 960))]
    r = row_image(frames)
    assert r.method == "stitched", r.note
    assert r.image.shape[1] > 900


def test_unrelated_frames_fall_back_to_sharpest_and_say_so():
    a, b = textured(seed=1), textured(seed=2)
    blurry = (a * 0.5 + 64).astype(np.uint8)
    r = row_image([("a", blurry), ("b", b)])
    assert r.method == "sharpest_frame_fallback" and r.source_frames == ["b"]
    assert "may undercount" in r.note


def test_empty_and_none_frames():
    with pytest.raises(ValueError):
        row_image([])
    with pytest.raises(ValueError):
        row_image([("x", None), ("y", np.zeros((0, 0, 3), np.uint8))])
