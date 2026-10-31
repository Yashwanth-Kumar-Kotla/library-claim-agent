"""Room measurement from a turn-around scan, against rooms rendered pixel by pixel."""

import numpy as np
import pytest

from libraryclaim import room_scan
from libraryclaim.room_scan import (ARM_M, CAMERA_HEIGHT_M, ScanFrame, item_size_cm, marker_scale, measure_room,
                                     rotation)

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


def scan(room, axis=(1.9, 1.4), cam_h=CAMERA_HEIGHT_M, step=5, scale_err=1.0, noise=0.0, tilt=True, upto=360, seed=0):
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


def test_depth_scale_error_without_marker_is_corrected_by_camera_height():
    frames, depth = scan(RECT, scale_err=1.10)
    r = measure_room(frames, depth)
    assert r.scale_factor == pytest.approx(1 / 1.10, rel=0.02)
    assert any("no marker seen" in n and "+/-10%" in n for n in r.notes)
    assert dims(r).floor_area_m2 == pytest.approx(13.02, rel=0.05)
    assert r.height_m == pytest.approx(2.5, abs=0.08)


@pytest.mark.parametrize("cam_h", [1.25, 1.55])
def test_short_or_tall_holder_stays_within_the_stated_error(cam_h):
    """The prior is a guess: a phone held 15 cm lower or higher than assumed is ~11% off in length."""
    frames, depth = scan(RECT, cam_h=cam_h, axis=(2.0, 1.5))
    r = measure_room(frames, depth)
    k = CAMERA_HEIGHT_M / cam_h
    assert dims(r).length_m == pytest.approx(4.2 * k, rel=0.04)
    assert abs(dims(r).length_m / 4.2 - 1) < 0.14


def test_no_floor_in_view_leaves_depth_scale_uncorrected_and_says_so():
    frames, depth = scan(RECT, scale_err=1.10, tilt=False, axis=(2.1, 1.55))
    r = measure_room(frames, depth)
    assert r.scale_factor == 1.0 and any("uncorrected" in n for n in r.notes)


def test_marker_in_a_frame_not_picked_for_its_heading_still_sets_the_scale(monkeypatch):
    frames, depth = scan(RECT, scale_err=1.10)
    marked = frames[7]                                     # one of many frames in its heading sector

    class M:
        side_px = 80
    monkeypatch.setattr(room_scan, "best_marker", lambda img: M() if img is marked.image else None)
    monkeypatch.setattr(room_scan, "marker_scale", lambda img, d, f, size: 1 / 1.10 if img is marked.image else None)
    r = measure_room(frames, depth)
    assert "printed marker" in r.method and r.scale_factor == pytest.approx(1 / 1.10, rel=0.001)
    assert any("camera-height check agrees" in n for n in r.notes)
    assert dims(r).floor_area_m2 == pytest.approx(13.02, rel=0.05)


@pytest.mark.parametrize("alpha,beta,gamma,heading,pitch", [
    (40, 90, 0, 40, 0), (299, 89.5, 67, 6, 0), (10, 130, 0, 10, 40), (200, 34, 0, 200, -56), (0, 90, -90, 270, 0)])
def test_aim_is_where_the_camera_points(alpha, beta, gamma, heading, pitch):
    from libraryclaim.room_scan import aim
    h, p = aim(alpha, beta, gamma)
    assert abs((h - heading + 180) % 360 - 180) < 1 and p == pytest.approx(pitch, abs=1)


def test_rolled_phone_frames_still_fill_the_circle():
    """The same turn reported with gamma soaking up part of the heading, as iOS does when upright."""
    frames, depth = scan(RECT)
    for n, f in enumerate(frames):
        if abs(f.beta - 90) < 1:
            g = ((n * 37) % 120) - 60                 # roll of -60..+60 degrees
            f.alpha, f.gamma = (f.alpha - g) % 360, g   # heading = alpha + gamma when upright
    r = measure_room(frames, depth)
    assert r.coverage_deg >= 337 and dims(r).floor_area_m2 == pytest.approx(13.02, rel=0.05)


def test_tilted_frames_with_their_own_depth_scale_are_aligned_to_the_walls():
    """Every frame tilted up reads 60% deep and every frame tilted down 20% shallow, as Depth Pro did
    at home (one depth map, one scale). Matching each frame's walls to the level frames undoes it."""
    frames, depth = scan(RECT)
    good = depth
    def depth(img):
        d, f = good(img)
        beta = frames[int(img[0, 0, 0]) + 256 * int(img[0, 0, 1])].beta
        return (d * 1.6 if beta > 110 else d * 0.8 if beta < 70 else d), f
    r = measure_room(frames, depth)
    assert r.height_m == pytest.approx(2.5, abs=0.1)
    assert dims(r).floor_area_m2 == pytest.approx(13.02, rel=0.06)


def test_one_bad_ceiling_frame_does_not_set_the_height():
    """The 4.34 m home reading: a single tilted-up frame whose depth is far off must not win."""
    frames, depth = scan(RECT)
    up = [n for n, f in enumerate(frames) if f.beta > 110]
    good = depth
    def depth(img):
        d, f = good(img)
        return (d * 1.6, f) if int(img[0, 0, 0]) + 256 * int(img[0, 0, 1]) == up[0] else (d, f)
    r = measure_room(frames, depth)
    assert r.height_m == pytest.approx(2.5, abs=0.08)


def test_furniture_dents_are_cleaned_but_a_real_l_shape_is_kept():
    from libraryclaim.room_scan import rectify
    t = np.linspace(0, 1, 50, endpoint=False)
    rect = np.concatenate([np.stack([4.2 * t, 0 * t], 1), np.stack([4.2 + 0 * t, 3.1 * t], 1),
                           np.stack([4.2 * (1 - t), 3.1 + 0 * t], 1), np.stack([0 * t, 3.1 * (1 - t)], 1)])
    dented = rect.copy()
    dented[60:66, 0] -= 0.3                                 # a 30 cm-wide chair in front of the right wall
    corners, note = rectify(dented)
    assert note.startswith("rectangle")
    from libraryclaim.room import room_figures
    assert room_figures(corners, 2.5).floor_area_m2 == pytest.approx(13.02, rel=0.03)


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
