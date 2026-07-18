"""Overlay grid geometry and homography mapping."""

import numpy as np
import pytest

from app.core.homography import homography_from_corners, map_points, patch_quads_image
from app.core.overlay import PRESETS, Overlay


def test_sg_preset_grid_shape():
    overlay = Overlay.from_preset(PRESETS[0])
    quads = overlay.patch_quads_unit()
    assert len(quads) == 8 * 12
    assert quads[0][:2] == (1, 1)
    assert quads[-1][:2] == (8, 12)
    # Row-major ordering — the export-order invariant.
    assert [q[:2] for q in quads[:3]] == [(1, 1), (1, 2), (1, 3)]


def test_patches_stay_inside_margins():
    overlay = Overlay(rows=2, cols=2, margin_x=10, margin_y=10, patch_size=50)
    for _, _, quad in overlay.patch_quads_unit():
        arr = np.asarray(quad)
        assert arr.min() >= 0.1 - 1e-9
        assert arr.max() <= 0.9 + 1e-9


def test_identity_homography_on_axis_aligned_quad():
    corners = [[0, 0], [100, 0], [100, 50], [0, 50]]
    H = homography_from_corners(corners)
    mapped = map_points(H, np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(mapped, [[50, 25], [0, 0], [100, 50]], atol=1e-9)


def test_perspective_quad_maps_corners_exactly():
    corners = [[12.0, 8.0], [205.0, 15.0], [198.0, 122.0], [5.0, 110.0]]
    H = homography_from_corners(corners)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    np.testing.assert_allclose(map_points(H, unit), corners, atol=1e-9)


def test_patch_quads_image_land_inside_chart():
    overlay = Overlay.from_preset(
        PRESETS[0], corners=[[10, 10], [250, 20], [240, 170], [5, 160]]
    )
    quads = patch_quads_image(overlay)
    assert len(quads) == 96
    all_pts = np.vstack([q for _, _, q in quads])
    assert all_pts[:, 0].min() > 5 and all_pts[:, 0].max() < 250
    assert all_pts[:, 1].min() > 10 and all_pts[:, 1].max() < 170


def test_degenerate_corners_raise():
    with pytest.raises(ValueError):
        homography_from_corners([[0, 0], [0, 0], [0, 0], [0, 0]])


def test_overlay_roundtrip_dict():
    overlay = Overlay.from_preset(PRESETS[0], corners=[[1, 2], [3, 4], [5, 6], [7, 8]])
    overlay.patch_offset = 1.5
    restored = Overlay.from_dict(overlay.to_dict())
    assert restored.to_dict() == overlay.to_dict()
