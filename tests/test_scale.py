import cv2
import numpy as np
import pytest

from libraryclaim.scale import (box_size_cm, find_markers, plane_from_marker, shelf_run_cm, spine_size)

PX_PER_CM = 10  # synthetic scene: marker of 15 cm drawn 150 px wide


def scene(marker_id=0, side_px=150, origin=(60, 300), size=(700, 1000)):
    img = np.full((*size, 3), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, side_px)
    x, y = origin
    img[y:y + side_px, x:x + side_px] = marker[..., None]
    return img


def warp(img, pts_src, pts_dst):
    H = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255)), H


def test_marker_found_and_scale_exact_frontal():
    markers = find_markers(scene())
    assert [m.id for m in markers] == [0]
    s = plane_from_marker(markers[0], 15.0)
    w, h = box_size_cm(s, (300, 100, 330, 320))  # 30 x 220 px
    assert w == pytest.approx(3.0, abs=0.1) and h == pytest.approx(22.0, abs=0.2)


def test_perspective_tilt_still_measures_within_tolerance():
    img = scene()
    h0, w0 = img.shape[:2]
    src = [[0, 0], [w0, 0], [w0, h0], [0, h0]]
    dst = [[0, 40], [w0, 0], [w0, h0], [0, h0 - 40]]  # phone turned ~15 degrees
    tilted, H = warp(img, src, dst)
    s = plane_from_marker(find_markers(tilted)[0], 15.0)
    corners = cv2.perspectiveTransform(np.float32([[[300, 100]], [[330, 320]]]), H).reshape(-1, 2)
    box = (*corners[0], *corners[1])
    size = spine_size(s, box)
    assert size.height_cm == pytest.approx(22.0, rel=0.05)
    assert size.thickness_cm == pytest.approx(3.0, rel=0.10)


def test_no_marker_returns_empty():
    assert find_markers(np.full((400, 400, 3), 200, np.uint8)) == []


def test_two_markers_both_reported():
    img = scene(0)
    other = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 3, 100)
    img[50:150, 700:800] = other[..., None]
    assert sorted(m.id for m in find_markers(img)) == [0, 3]


def test_small_marker_flagged_unreliable():
    s = plane_from_marker(find_markers(scene(side_px=36))[0], 15.0)
    assert not s.reliable


def test_flat_book_height_is_its_long_side():
    s = plane_from_marker(find_markers(scene())[0], 15.0)
    size = spine_size(s, (300, 400, 530, 430))  # 23 cm wide, 3 cm tall
    assert size.lying_flat and size.height_cm == pytest.approx(23.0, abs=0.2)
    assert size.thickness_cm == pytest.approx(3.0, abs=0.1)


def test_detector_orientation_overrides_box_shape():
    s = plane_from_marker(find_markers(scene())[0], 15.0)
    # Square-ish box for a thick upright book: trust the detector's "upright".
    size = spine_size(s, (300, 100, 420, 280), lying_flat=False)
    assert not size.lying_flat and size.height_cm == pytest.approx(18.0, abs=0.2)


def test_implausible_sizes_flagged():
    s = plane_from_marker(find_markers(scene())[0], 15.0)
    assert not spine_size(s, (300, 100, 302, 160)).plausible   # 0.2 x 6 cm: a crease, not a book
    assert not spine_size(s, (300, 0, 330, 690)).plausible     # 69 cm tall


def test_wrong_marker_size_rejected():
    with pytest.raises(ValueError):
        plane_from_marker(find_markers(scene())[0], 0)


def test_shelf_run_unions_overlaps_and_skips_gaps():
    s = plane_from_marker(find_markers(scene())[0], 15.0)
    boxes = [
        (300, 100, 330, 320),   # 3 cm
        (330, 100, 350, 320),   # 2 cm, touching
        (345, 280, 575, 320),   # flat book under/overlapping: extends run to 57.5 -> 27.5 cm total from 30
        (700, 100, 740, 320),   # 4 cm after a gap
    ]
    assert shelf_run_cm(s, boxes) == pytest.approx(27.5 + 4.0, abs=0.3)
    assert shelf_run_cm(s, []) == 0


@pytest.mark.parametrize("quiet", [0, 5, 10, 15, 25])
@pytest.mark.parametrize("seed", range(4))
def test_marker_found_on_textured_shelf_any_quiet_zone(quiet, seed):
    img = np.random.default_rng(seed).normal(150, 6, (420, 600, 3)).clip(0, 255).astype(np.uint8)
    img[250 - quiet:400 + quiet, 20 - min(quiet, 20):170 + quiet] = 255
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 150)
    img[250:400, 20:170] = m[..., None]
    assert [x.id for x in find_markers(img)] == [0]


def test_marker_cut_by_frame_edge_not_reported():
    img = scene(origin=(0, 300))[:, 40:]  # left part of the marker cropped away
    assert find_markers(img) == []
