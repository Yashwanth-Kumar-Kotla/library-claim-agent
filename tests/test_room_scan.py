"""Room measurement from a turn-around scan, against rooms rendered pixel by pixel."""

import numpy as np
import pytest

from libraryclaim.room_scan import (ARM_M, ScanFrame, item_size_cm, marker_scale, measure_room, rotation)

W, H, F = 108, 192, 158.0   # portrait frame, ~62 degree vertical field of view like a phone


def render(alpha, beta, room, axis, cam_h, arm=ARM_M):
    """Depth (along the optical axis) of every pixel for a camera inside an extruded floor polygon."""
    R = rotation(alpha, beta, 0)
    fwd = R @ np.array([0, 0, -1.0])
    fh = np.array([fwd[0], fwd[1], 0])
    cam = np.array([axis[0], axis[1], cam_h]) + arm * fh / np.linalg.norm(fh)
    v, u = np.mgrid[0:H, 0:W]
    dirs = np.stack([(u - W / 2) / F, -(v - H / 2) / F, -np.ones_like(u, float)], -1) @ R.T
    t = np.full((H, W), np.inf)
    height = room["height"]
    for zplane in (0.0, height):  # floor and ceiling
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = (zplane - cam[2]) / dirs[..., 2]
        t = np.where((tz > 1e-6) & (tz < t), tz, t)
    pts = room["corners"]
    for i in range(len(pts)):  # walls: segments of the outline
        a, b = np.array(pts[i], float), np.array(pts[(i + 1) % len(pts)], float)
        e = b - a
        denom = dirs[..., 0] * e[1] - dirs[..., 1] * e[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            tw = ((a[0] - cam[0]) * e[1] - (a[1] - cam[1]) * e[0]) / denom
            sw = ((a[0] - cam[0]) * dirs[..., 1] - (a[1] - cam[1]) * dirs[..., 0]) / denom
        t = np.where((tw > 1e-6) & (sw >= 0) & (sw <= 1) & (tw < t), tw, t)
    for lo, hi in room.get("boxes", []):  # furniture: axis-aligned boxes (slab test)
        lo, hi = np.array(lo, float), np.array(hi, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - cam) / dirs, (hi - cam) / dirs
        tn = np.nanmax(np.minimum(t1, t2), axis=-1)
        tf = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tn <= tf) & (tn > 1e-6)
        t = np.where(hit & (tn < t), tn, t)
    return t.astype(np.float32)  # t is depth along the axis: the device-frame direction has z = -1


def scan(room, axis=(1.9, 1.4), cam_h=1.45, step=5, scale_err=1.0, noise=0.0, tilt=True, upto=360, seed=0):
    rng = np.random.default_rng(seed)
    frames, depths = [], {}
    def add(alpha, beta):
        img = np.full((H, W, 3), 128, np.uint8)
        img[0, 0] = [len(frames) % 256, len(frames) // 256, 0]
        img[1:, :] = rng.integers(0, 255, (H - 1, W, 3), dtype=np.uint8)  # texture for sharpness
        d = render(alpha, beta, room, axis, cam_h) * scale_err
        if noise:
            d = d * (1 + rng.normal(0, noise, d.shape))
        depths[len(frames)] = d
        frames.append(ScanFrame(img, alpha, beta, 0.0))
    for a in range(0, upto, step):
        add(a, 90)
    if tilt:
        for a in (0, 120, 240):
            add(a, 130)   # tilted up toward the ceiling
            add(a, 50)    # tilted down toward the floor
    def depth(img):
        return depths[int(img[0, 0, 0]) + 256 * int(img[0, 0, 1])], F
    return frames, depth


RECT = {"corners": [(0, 0), (4.2, 0), (4.2, 3.1), (0, 3.1)], "height": 2.5}


def dims(result):
    from libraryclaim.room import room_figures
    return room_figures(result.corners, result.height_m or 2.5)


def test_rectangular_room_measured_within_tolerance():
    frames, depth = scan(RECT)
    r = measure_room(frames, depth)
    fig = dims(r)
    assert fig.floor_area_m2 == pytest.approx(4.2 * 3.1, rel=0.05)
    assert fig.length_m == pytest.approx(4.2, rel=0.04) and fig.width_m == pytest.approx(3.1, rel=0.04)
    assert r.height_m == pytest.approx(2.5, abs=0.08)
    assert fig.shape_note == "rectangular"


def test_rotated_room_and_off_centre_standing_point():
    c, s = np.cos(np.radians(25)), np.sin(np.radians(25))
    room = {"corners": [(x * c - y * s, x * s + y * c) for x, y in RECT["corners"]], "height": 2.6}
    frames, depth = scan(room, axis=(0.9 * c - 1.0 * s, 0.9 * s + 1.0 * c))
    fig = dims(measure_room(frames, depth))
    assert fig.floor_area_m2 == pytest.approx(13.02, rel=0.06)


def test_noisy_depth_still_close():
    frames, depth = scan(RECT, noise=0.03)
    assert dims(measure_room(frames, depth)).floor_area_m2 == pytest.approx(13.02, rel=0.07)


def test_l_shaped_room_is_not_forced_into_a_rectangle():
    L = {"corners": [(0, 0), (5, 0), (5, 2.2), (3, 2.2), (3, 4), (0, 4)], "height": 2.5}
    frames, depth = scan(L, axis=(1.6, 1.4))
    r = measure_room(frames, depth)
    assert "non-rectangular" in " ".join(r.notes)
    assert dims(r).floor_area_m2 == pytest.approx(5 * 2.2 + 3 * 1.8, rel=0.12)


def test_depth_scale_error_is_reported_when_uncorrected():
    frames, depth = scan(RECT, scale_err=1.10)
    r = measure_room(frames, depth)
    assert r.scale_factor == 1.0 and any("no marker" in n for n in r.notes)
    assert dims(r).floor_area_m2 == pytest.approx(13.02 * 1.21, rel=0.07)  # the 10% shows up, area ~21%


def test_marker_scale_corrects_a_10_percent_depth_error():
    import cv2
    img = np.full((600, 600, 3), 255, np.uint8)
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 120)
    img[240:360, 240:360] = m[..., None]
    f = 600.0
    true_z = f * 0.15 / 120          # 0.75 m: a 15 cm marker spanning 120 px
    depth = np.full((600, 600), true_z * 1.10, np.float32)
    assert marker_scale(img, depth, f, 0.15) == pytest.approx(1 / 1.10, rel=0.03)
    assert marker_scale(np.full((600, 600, 3), 200, np.uint8), depth, f, 0.15) is None


def test_incomplete_turn_is_refused():
    frames, depth = scan(RECT, upto=200)
    with pytest.raises(ValueError, match="covered"):
        measure_room(frames, depth)


def test_no_tilt_means_no_height_rather_than_a_guess():
    frames, depth = scan(RECT, tilt=False, axis=(2.1, 1.55))
    r = measure_room(frames, depth)
    assert r.height_m is None and any("height" in n for n in r.notes)
    assert dims(r).floor_area_m2 == pytest.approx(13.02, rel=0.08)


def test_item_front_face_size():
    depth = np.full((400, 400), 2.0, np.float32)
    w, h = item_size_cm((100, 100, 200, 300), depth, focal=400.0)
    assert (w, h) == (50, 100)
    assert item_size_cm((0, 0, 2, 2), depth, 400.0) is None


def test_furniture_does_not_shrink_the_room_or_fake_the_floor():
    room = dict(RECT, boxes=[((1.2, 1.9, 0.0), (2.4, 2.6, 0.75)),     # desk in the room, top at 0.75 m
                             ((0.0, 0.5, 0.0), (0.35, 2.3, 2.0))])    # bookcase against the left wall
    frames, depth = scan(room, axis=(2.6, 1.2))
    r = measure_room(frames, depth)
    assert r.height_m == pytest.approx(2.5, abs=0.08)               # floor, not the desk top
    assert dims(r).floor_area_m2 == pytest.approx(13.02, rel=0.07)   # wall behind the bookcase still found
