"""Sampling reads raw values exactly; means match numpy ground truth."""

import numpy as np

from app.core.overlay import Overlay
from app.core.sampler import sample_overlay


def _uniform_patch_frame():
    """A 2x3 chart where every patch region has a single known float value,
    including scene-referred values outside [0, 1]."""
    frame = np.full((120, 180, 3), 99.0, dtype=np.float32)  # poison background
    values = {}
    rows, cols = 2, 3
    for r in range(rows):
        for c in range(cols):
            value = np.array(
                [r + c + 0.125, (r + 1) * 0.5, -0.25 + c], dtype=np.float32
            )
            y0, y1 = r * 60, (r + 1) * 60
            x0, x1 = c * 60, (c + 1) * 60
            frame[y0:y1, x0:x1] = value
            values[(r + 1, c + 1)] = value
    return frame, values


def test_uniform_patches_sample_exact_values():
    frame, values = _uniform_patch_frame()
    overlay = Overlay(
        rows=2, cols=3, margin_x=0, margin_y=0, patch_size=50,
        corners=[[0, 0], [180, 0], [180, 120], [0, 120]],
    )
    samples = sample_overlay(frame, overlay)
    assert len(samples) == 6
    for sample in samples:
        expected = values[(sample.row, sample.col)]
        # Uniform region: the mean must equal the value exactly.
        np.testing.assert_array_equal(np.asarray(sample.rgb, dtype=np.float32), expected)
        assert sample.pixel_count > 400  # 50% of a 60x60 cell ≈ 30x30 px


def test_mean_matches_numpy_on_nonuniform_data():
    rng = np.random.default_rng(11)
    frame = rng.standard_normal((80, 80, 3)).astype(np.float32) * 3.0
    overlay = Overlay(
        rows=1, cols=1, margin_x=0, margin_y=0, patch_size=100,
        corners=[[20, 20], [60, 20], [60, 60], [20, 60]],
    )
    (sample,) = sample_overlay(frame, overlay)
    region = frame[20:60, 20:60].astype(np.float64).reshape(-1, 3)
    np.testing.assert_allclose(sample.rgb, region.mean(axis=0), rtol=0, atol=1e-12)


def test_out_of_image_patch_returns_nan_not_garbage():
    frame = np.ones((50, 50, 3), dtype=np.float32)
    overlay = Overlay(
        rows=1, cols=1, margin_x=0, margin_y=0, patch_size=100,
        corners=[[200, 200], [300, 200], [300, 300], [200, 300]],
    )
    (sample,) = sample_overlay(frame, overlay)
    assert sample.pixel_count == 0
    assert all(np.isnan(v) for v in sample.rgb)


def test_row_major_order():
    frame = np.zeros((60, 60, 3), dtype=np.float32)
    overlay = Overlay(
        rows=2, cols=2, margin_x=0, margin_y=0, patch_size=50,
        corners=[[0, 0], [60, 0], [60, 60], [0, 60]],
    )
    samples = sample_overlay(frame, overlay)
    assert [(s.row, s.col) for s in samples] == [(1, 1), (1, 2), (2, 1), (2, 2)]
