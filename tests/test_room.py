import pytest

from libraryclaim.room import M2_TO_FT2, Opening, room_figures

RECT = [(0, 0), (4, 0), (4, 3), (0, 3)]


def test_rectangle():
    r = room_figures(RECT, 2.5)
    assert (r.length_m, r.width_m) == (4, 3)
    assert r.floor_area_m2 == pytest.approx(12)
    assert r.wall_area_m2 == pytest.approx(14 * 2.5)
    assert r.shape_note == "rectangular"
    f = r.as_room_fields()
    assert f["floor_area_ft2"] == pytest.approx(12 * M2_TO_FT2, abs=0.1)


def test_order_and_orientation_do_not_matter():
    a = room_figures(RECT, 2.5)
    b = room_figures(list(reversed(RECT)), 2.5)
    c = room_figures(RECT[2:] + RECT[:2], 2.5)
    assert a.floor_area_m2 == b.floor_area_m2 == c.floor_area_m2


def test_closed_polygon_duplicate_and_collinear_points():
    messy = [(0, 0), (2, 0), (4, 0), (4, 0), (4, 3), (0, 3), (0, 0)]
    r = room_figures(messy, 2.5)
    assert r.shape_note == "rectangular" and r.floor_area_m2 == pytest.approx(12)


def test_l_shaped_room():
    L = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 4), (0, 4)]
    r = room_figures(L, 2.4)
    assert r.floor_area_m2 == pytest.approx(5 * 2 + 3 * 2)
    assert r.wall_area_m2 == pytest.approx(18 * 2.4)
    assert "non-rectangular" in r.shape_note
    assert (r.length_m, r.width_m) == pytest.approx((5, 4))


def test_slightly_skewed_rectangle_still_rectangle():
    # Real measurements are never perfectly square.
    r = room_figures([(0, 0), (4, 0.05), (3.98, 3.02), (0.02, 3)], 2.5)
    assert r.shape_note == "rectangular"


def test_rotated_rectangle_dims():
    r = room_figures([(0, 0), (2.83, 2.83), (0.71, 4.95), (-2.12, 2.12)], 2.5)
    assert r.length_m == pytest.approx(4.0, abs=0.01) and r.width_m == pytest.approx(3.0, abs=0.01)


def test_openings_and_shelving():
    r = room_figures(RECT, 2.5, openings=[Opening(0.9, 2.1)], shelving=[(1.8, 2.0), (0.9, 2.0)])
    assert r.wall_area_net_m2 == pytest.approx(35 - 1.89)
    assert r.wall_area_m2 == pytest.approx(35)  # gross is the reported figure
    assert r.shelved_wall_area_m2 == pytest.approx(5.4)
    assert not r.warnings


def test_shelf_taller_than_ceiling_warns():
    r = room_figures(RECT, 2.4, shelving=[(1, 2.6)])
    assert any("taller than the ceiling" in w for w in r.warnings)


def test_unusual_height_warns_but_computes():
    assert room_figures(RECT, 1.2).warnings


@pytest.mark.parametrize("corners,height", [
    ([(0, 0), (1, 0)], 2.5),                           # two points
    ([(0, 0), (1, 0), (2, 0), (3, 0)], 2.5),           # all collinear
    ([(0, 0), (1, 1), (1, 1)], 2.5),                   # degenerate
    (RECT, 0),
    (RECT, -1),
    ([(0, 0), (4, 3), (4, 0), (0, 3)], 2.5),           # bow-tie: corners out of order
])
def test_bad_inputs_rejected(corners, height):
    with pytest.raises(ValueError):
        room_figures(corners, height)
