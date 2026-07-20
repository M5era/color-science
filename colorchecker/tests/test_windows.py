"""Smooth mask windows + the Reuleaux Fine zone stage built on them."""

import numpy as np

from app.core.reuleaux import rgb_to_reuleaux
from app.core.stages import ReuleauxFineStage
from app.core.windows import plateau_window, wrapped_window


# ------------------------------------------------------------- windows

def test_plateau_core_shoulder_and_zero():
    x = np.linspace(-1.0, 2.0, 3001)
    w = plateau_window(x, center=0.5, flat=0.2, soft=0.1)
    assert (w[np.abs(x - 0.5) <= 0.2 - 1e-9] == 1.0).all()
    assert (w[np.abs(x - 0.5) >= 0.3 + 1e-9] == 0.0).all()
    assert ((w >= 0.0) & (w <= 1.0)).all()


def test_plateau_is_smooth_no_kinks():
    # C1: numerical derivative must be continuous (no jumps) across
    # the flat->shoulder and shoulder->zero transitions
    x = np.linspace(0.0, 1.0, 200001)
    w = plateau_window(x, center=0.5, flat=0.1, soft=0.15)
    d = np.diff(w)
    assert np.abs(np.diff(d)).max() < 1e-8


def test_plateau_falloff_monotone():
    x = np.linspace(0.5, 1.0, 1000)  # walking away from the center
    w = plateau_window(x, center=0.5, flat=0.1, soft=0.2)
    assert (np.diff(w) <= 1e-12).all()


def test_plateau_wide_core_covers_domain():
    x = np.linspace(0.0, 1.0, 101)
    np.testing.assert_array_equal(
        plateau_window(x, center=0.5, flat=2.0, soft=0.25), 1.0
    )


def test_wrapped_window_crosses_the_seam():
    # a window centered just below 1.0 must reach hues just above 0.0
    w = wrapped_window(np.array([0.02, 0.5]), center=0.97, flat=0.06, soft=0.05)
    assert w[0] == 1.0          # 0.02 is 0.05 away around the seam
    assert w[1] == 0.0          # 0.5 is on the far side of the wheel


def test_wrapped_window_uses_shorter_arc():
    w_near = wrapped_window(np.array([0.9]), center=0.1, flat=0.0, soft=0.3)
    w_far = wrapped_window(np.array([0.5]), center=0.1, flat=0.0, soft=0.3)
    assert w_near[0] > 0.0      # 0.2 away around the seam
    assert w_far[0] == 0.0      # 0.4 away, outside


# ---------------------------------------------------- Reuleaux Fine

def _params(stage, **overrides):
    """Identity params with named overrides (index map from docstring)."""
    idx = {"hue_center": 0, "hue_flat": 1, "hue_soft": 2, "hue_shift": 3,
           "sat_adj": 4, "val_adj": 5, "luma_center": 6, "luma_flat": 7,
           "luma_soft": 8, "sat_center": 9, "sat_flat": 10, "sat_soft": 11}
    p = stage.identity().copy()
    for key, value in overrides.items():
        p[idx[key]] = value
    return p


def test_fine_zone_is_local_in_hue():
    stage = ReuleauxFineStage()
    reddish = np.array([[0.8, 0.3, 0.25]])
    bluish = np.array([[0.25, 0.3, 0.8]])
    hue_red = rgb_to_reuleaux(reddish)[0, 0]

    p = _params(stage, hue_center=hue_red, hue_flat=0.05, hue_soft=0.08,
                sat_adj=1.6, val_adj=0.4, hue_shift=0.03)
    assert np.abs(stage.apply(reddish, p) - reddish).max() > 1e-3
    np.testing.assert_allclose(stage.apply(bluish, p), bluish, atol=1e-12)


def test_fine_zone_protects_neutrals():
    stage = ReuleauxFineStage()
    grays = np.array([[0.18, 0.18, 0.18], [0.7, 0.7, 0.7]])
    hue_gray = rgb_to_reuleaux(grays)[0, 0]
    p = _params(stage, hue_center=hue_gray, hue_flat=0.2, hue_soft=0.2,
                sat_adj=1.9, val_adj=2.0, hue_shift=0.1)
    np.testing.assert_allclose(stage.apply(grays, p), grays, atol=1e-9)


def test_fine_luma_mask_gates_the_zone():
    stage = ReuleauxFineStage()
    dark = np.array([[0.16, 0.06, 0.05]])
    bright = np.array([[0.8, 0.3, 0.25]])
    hue = rgb_to_reuleaux(bright)[0, 0]

    # same hue zone, but luma mask restricted to shadows
    p = _params(stage, hue_center=hue, hue_flat=0.1, hue_soft=0.1,
                sat_adj=1.7, luma_center=0.1, luma_flat=0.1, luma_soft=0.15)
    assert np.abs(stage.apply(dark, p) - dark).max() > 1e-4
    np.testing.assert_allclose(stage.apply(bright, p), bright, atol=1e-12)


def test_fine_sat_mask_gates_the_zone():
    stage = ReuleauxFineStage()
    saturated = np.array([[0.8, 0.15, 0.1]])
    muted = np.array([[0.5, 0.42, 0.4]])
    hue_sat = rgb_to_reuleaux(saturated)[0]
    hue_mut = rgb_to_reuleaux(muted)[0]
    assert abs(hue_sat[0] - hue_mut[0]) < 0.05  # same hue family

    # restrict to high saturation only
    p = _params(stage, hue_center=hue_sat[0], hue_flat=0.1, hue_soft=0.1,
                val_adj=0.8, sat_center=hue_sat[1], sat_flat=0.08,
                sat_soft=0.1)
    assert np.abs(stage.apply(saturated, p) - saturated).max() > 1e-3
    np.testing.assert_allclose(stage.apply(muted, p), muted, atol=1e-12)


def test_fine_hue_center_wraps():
    stage = ReuleauxFineStage()
    reddish = np.array([[0.8, 0.3, 0.25]])
    hue = rgb_to_reuleaux(reddish)[0, 0]
    # center expressed one full turn below: identical zone
    p_a = _params(stage, hue_center=hue, hue_flat=0.05, hue_soft=0.08,
                  sat_adj=1.5)
    p_b = p_a.copy()
    p_b[0] = hue - 1.0
    np.testing.assert_allclose(
        stage.apply(reddish, p_a), stage.apply(reddish, p_b), atol=1e-12
    )


def test_fine_describe_reports_zone_and_masks():
    stage = ReuleauxFineStage()
    text = stage.describe(stage.identity())
    assert "Hue center" in text
    assert text.count("wide open") == 2  # both masks off at identity

    p = _params(stage, luma_center=0.2, luma_flat=0.1, luma_soft=0.1)
    assert stage.describe(p).count("wide open") == 1
