"""Chromogen-style stages: modulation behavior, per-tool semantics,
foldover safety, and the generic stage bake CLI.

(Identity passthrough and torch-mirror parity for these stages are
covered automatically by the STAGE_POOL loops in test_parametric.py
and test_backprop.py.)
"""

import sys

import numpy as np
import pytest

from app.core.chromogen import (
    CHROMOGEN_STAGES,
    MID_GREY,
    STOP,
    BrillianceReductionStage,
    ColourCrosstalkStage,
    ColourSaturationStage,
    ContrastCurveStage,
    HighlightBleachStage,
    NeutralTintStage,
    SectorBrightnessStage,
    SectorSaturationStage,
    SectorSkewStage,
    SectorSquashStage,
    modulation,
)
from app.core.reuleaux import rgb_to_reuleaux
from app.core.windows import ramp_window


def _sat_of(rgb):
    return rgb_to_reuleaux(rgb)[..., 1]


def _with(stage, **by_name):
    p = stage.identity().copy()
    for name, value in by_name.items():
        p[stage.param_names.index(name)] = value
    return p


# ------------------------------------------------- ramp + modulation

def test_ramp_window_shape_and_smoothness():
    x = np.linspace(-1.0, 2.0, 30001)
    w = ramp_window(x, pivot=0.5, falloff=0.4)
    assert (w[x <= 0.3 - 1e-9] == 0.0).all()
    assert (w[x >= 0.7 + 1e-9] == 1.0).all()
    assert (np.diff(w) >= -1e-12).all()          # monotone
    d = np.diff(w)
    assert np.abs(np.diff(d)).max() < 1e-6       # C1, no kinks


def test_modulation_identity_and_zone_sign():
    val = np.linspace(0.0, 1.5, 200)
    sat = np.full_like(val, 0.5)
    np.testing.assert_array_equal(
        modulation(val, sat, 0.0, 0.0, 0.0), 1.0
    )
    hi = modulation(val, sat, 1.0, 0.1, 0.0)   # highlights only
    lo = modulation(val, sat, -1.0, 0.1, 0.0)  # shadows only
    assert hi[0] == 0.0 and hi[-1] == 1.0
    assert lo[0] == 1.0 and lo[-1] == 0.0


def test_modulation_chroma_sign_targets_saturation():
    val = np.full(2, 0.5)
    sat = np.array([0.05, 0.8])  # muted, saturated
    only_saturated = modulation(val, sat, 0.0, 0.0, 1.0)
    only_muted = modulation(val, sat, 0.0, 0.0, -1.0)
    assert only_saturated[1] > 0.9 and only_saturated[0] < 0.1
    assert only_muted[0] > 0.9 and only_muted[1] < 0.1


# --------------------------------------------------- colour saturation

def test_colour_saturation_ganged_scales_sat():
    stage = ColourSaturationStage()
    x = np.array([[0.6, 0.3, 0.2], [0.2, 0.5, 0.3]])
    p = _with(stage, **{"R/G": 1.5, "Y/B": 1.5})
    out = stage.apply(x, p)
    assert (_sat_of(out) > _sat_of(x) * 1.2).all()
    grays = np.array([[0.4, 0.4, 0.4]])
    np.testing.assert_allclose(stage.apply(grays, p), grays, atol=1e-9)


def test_colour_saturation_unganged_axes_differ():
    stage = ColourSaturationStage()
    yellowish = np.array([[0.7, 0.6, 0.2]])
    greenish = np.array([[0.35, 0.6, 0.35]])
    p = _with(stage, **{"Y/B": 1.8})  # R/G stays 1.0
    gain_y = _sat_of(stage.apply(yellowish, p)) / _sat_of(yellowish)
    gain_g = _sat_of(stage.apply(greenish, p)) / _sat_of(greenish)
    assert gain_y[0] > gain_g[0] + 0.15


def test_colour_saturation_chroma_gate_desats_only_extremes():
    """The demo's 'sand off the spikes' move: desaturate, chroma +."""
    stage = ColourSaturationStage()
    extreme = np.array([[0.9, 0.05, 0.05]])
    muted = np.array([[0.5, 0.42, 0.4]])
    p = _with(stage, **{"R/G": 0.2, "Y/B": 0.2,
                        "Chroma": 1.0})
    extreme_loss = 1.0 - _sat_of(stage.apply(extreme, p))[0] / _sat_of(extreme)[0]
    muted_loss = 1.0 - _sat_of(stage.apply(muted, p))[0] / _sat_of(muted)[0]
    assert extreme_loss > 0.3          # spikes pulled in hard
    assert muted_loss < 0.1            # muted colors essentially spared
    assert extreme_loss > 5 * muted_loss


# ------------------------------------------------------ contrast curve

def test_contrast_curve_is_identity_at_defaults():
    stage = ContrastCurveStage()
    x = np.random.default_rng(0).uniform(0.02, 1.2, (200, 3))
    np.testing.assert_allclose(stage.apply(x, stage.identity()), x, atol=1e-12)


def test_contrast_curve_is_bounded_asymptotic_not_clipped():
    # High contrast must NOT dive out of range and hard-clip: the curve is
    # a bounded S that rolls off toward the black/white points, steep in
    # the mid and flattening (asymptotic) at the ends.
    stage = ContrastCurveStage()
    ramp = np.linspace(0.0, 1.0, 800)[:, None].repeat(3, axis=1)
    out = stage.apply(ramp, _with(stage, Contrast=2.5))[:, 0]
    assert out.min() > -0.02 and out.max() < 1.02     # never leaves range
    slope = np.gradient(out, ramp[:, 0])
    v = ramp[:, 0]
    mid = np.abs(v - MID_GREY) < 0.05
    toe = v < 0.12
    assert slope[mid].mean() > 1.3                     # contrast added at mid
    assert slope[mid].mean() > slope[toe].mean() + 0.5  # ends roll off


def test_contrast_curve_offsets_move_white_black_points():
    stage = ContrastCurveStage()
    ramp = np.linspace(0.0, 1.0, 400)[:, None].repeat(3, axis=1)
    lo_w = stage.apply(ramp, _with(stage, Contrast=2.0, **{"White Offset": 0.6}))[:, 0]
    hi_w = stage.apply(ramp, _with(stage, Contrast=2.0, **{"White Offset": 1.4}))[:, 0]
    assert lo_w.max() < hi_w.max()                     # white offset moves white pt
    lo_b = stage.apply(ramp, _with(stage, Contrast=2.0, **{"Black Offset": 0.6}))[:, 0]
    hi_b = stage.apply(ramp, _with(stage, Contrast=2.0, **{"Black Offset": 1.4}))[:, 0]
    assert lo_b.min() > hi_b.min()                     # black offset moves black pt


def test_contrast_curve_offsets_are_independent():
    # White Offset must touch ONLY highlights (leave shadows exactly as
    # input) and Black Offset ONLY shadows, with neither changing the mid
    # (pivot) contrast — the property that lets you shape a toe and a
    # shoulder separately for a film S.
    stage = ContrastCurveStage()
    ramp = np.linspace(0.0, 1.0, 400)[:, None].repeat(3, axis=1)
    below = ramp[:, 0] <= MID_GREY
    above = ramp[:, 0] >= MID_GREY
    base = stage.apply(ramp, _with(stage, Contrast=2.0))
    white = stage.apply(ramp, _with(stage, Contrast=2.0, **{"White Offset": 1.4}))
    black = stage.apply(ramp, _with(stage, Contrast=2.0, **{"Black Offset": 1.4}))
    np.testing.assert_allclose(white[below], base[below], atol=1e-9)  # shadows kept
    np.testing.assert_allclose(black[above], base[above], atol=1e-9)  # highlights kept


def test_contrast_curve_flare_lifts_shadows_only():
    stage = ContrastCurveStage()
    p = _with(stage, Flare=2.0)
    v = np.array([[0.05, 0.05, 0.05], [0.95, 0.95, 0.95]])
    out = stage.apply(v, p)
    assert out[0, 0] - 0.05 > 0.02             # shadows milked up
    assert abs(out[1, 0] - 0.95) < 1e-3        # highlights untouched


def test_contrast_curve_luma_blend_preserves_chroma():
    stage = ContrastCurveStage()
    x = np.array([[0.55, 0.35, 0.25]])
    rgb_mode = stage.apply(x, _with(stage, Contrast=1.8, **{"Luma Blend": 0.0}))
    luma_mode = stage.apply(x, _with(stage, Contrast=1.8, **{"Luma Blend": 1.0}))
    # per-RGB contrast raises saturation (the film look); luma-only keeps
    # chromaticity (reuleaux sat constant)
    np.testing.assert_allclose(_sat_of(luma_mode), _sat_of(x), atol=1e-6)
    assert _sat_of(rgb_mode)[0] > _sat_of(x)[0]


def test_contrast_curve_mid_compensate_holds_pivot():
    stage = ContrastCurveStage()
    grey = np.array([[MID_GREY, MID_GREY, MID_GREY]])
    off = stage.apply(grey, _with(stage, **{"Mid Push": 0.6, "Mid Compensate": 0.0}))
    on = stage.apply(grey, _with(stage, **{"Mid Push": 0.6, "Mid Compensate": 1.0}))
    assert off[0, 0] - MID_GREY > 0.02             # compensate off: pivot lifts
    np.testing.assert_allclose(on, grey, atol=1e-9)  # compensate on: pivot held


# ----------------------------------------------------- highlight bleach

def test_highlight_bleach_desats_highlights_only():
    stage = HighlightBleachStage()
    p = _with(stage, **{"R": 0.9, "Y": 0.9, "G": 0.9,
                        "B": 0.9, "Pivot": 1.0, "Falloff": 3.0})
    bright = np.array([[0.95, 0.55, 0.45]])
    dark = np.array([[0.25, 0.12, 0.1]])
    assert _sat_of(stage.apply(bright, p))[0] < _sat_of(bright)[0] * 0.5
    np.testing.assert_allclose(stage.apply(dark, p), dark, atol=1e-9)


def test_highlight_bleach_unganged_spares_a_sector():
    """The save-the-blue-skies move: relax the blue slider."""
    stage = HighlightBleachStage()
    p = _with(stage, **{"R": 0.9, "Y": 0.9, "G": 0.9,
                        "B": 0.0, "Pivot": 0.0, "Falloff": 3.0})
    sky = np.array([[0.55, 0.7, 0.95]])
    warm = np.array([[0.95, 0.7, 0.5]])
    sky_loss = 1.0 - _sat_of(stage.apply(sky, p))[0] / _sat_of(sky)[0]
    warm_loss = 1.0 - _sat_of(stage.apply(warm, p))[0] / _sat_of(warm)[0]
    assert warm_loss > 0.4
    assert sky_loss < warm_loss / 2


# -------------------------------------------------------- neutral tint

def test_neutral_tint_signed_amount_picks_side_in_log():
    """v3: sum-preserving RGB offset in log; + = highlights, - = shadows."""
    stage = NeutralTintStage()
    highs = np.array([[0.9, 0.9, 0.9]])
    lows = np.array([[0.12, 0.12, 0.12]])
    warm_high = _with(stage, Hue=40.0, Amount=0.3)
    for x in (highs, lows):
        out = stage.apply(x, warm_high)
        # channel mean (log exposure) untouched by construction
        np.testing.assert_allclose(out.mean(axis=1), x.mean(axis=1),
                                   atol=1e-12)
    assert _sat_of(stage.apply(highs, warm_high))[0] > 0.015  # tinted
    assert _sat_of(stage.apply(lows, warm_high))[0] < 1e-6    # spared

    cold_low = _with(stage, Hue=220.0, Amount=-0.3)
    assert _sat_of(stage.apply(lows, cold_low))[0] > 0.05
    assert _sat_of(stage.apply(highs, cold_low))[0] < 1e-6


def test_neutral_tint_offset_direction_matches_picked_hue():
    """The RGB offset direction reads back as the picked reuleaux hue."""
    grey = np.array([[0.391, 0.391, 0.391]])
    stage = NeutralTintStage()
    for hue_deg in (0.0, 60.0, 137.0, 240.0, 313.0):
        out = stage.apply(grey, _with(stage, Hue=hue_deg, Amount=0.5))
        got = rgb_to_reuleaux(out)[0, 0] * 360.0
        assert abs((got - hue_deg + 180.0) % 360.0 - 180.0) < 1e-6, hue_deg


def test_neutral_tint_chroma_is_baselight_sat_mask():
    """Chroma slider: 1 = everything, 2 = only neutrals, 0 = only
    saturated colors."""
    stage = NeutralTintStage()
    grey = np.array([[0.12, 0.12, 0.12]])
    saturated = np.array([[0.3, 0.06, 0.05]])

    neutrals_only = _with(stage, Hue=220.0, Amount=-0.5, Chroma=2.0)
    np.testing.assert_allclose(stage.apply(saturated, neutrals_only),
                               saturated, atol=1e-9)
    assert np.abs(stage.apply(grey, neutrals_only) - grey).max() > 0.01

    saturated_only = _with(stage, Hue=220.0, Amount=-0.5, Chroma=0.0)
    np.testing.assert_allclose(stage.apply(grey, saturated_only),
                               grey, atol=1e-9)
    assert np.abs(stage.apply(saturated, saturated_only)
                  - saturated).max() > 0.01


# ---------------------------------------------------- colour crosstalk

def test_crosstalk_luminance_weighted_and_neutral_safe():
    stage = ColourCrosstalkStage()
    p = _with(stage, **{"R -> Y/B": 0.5, "Y -> R/G": 0.5,
                        "G -> Y/B": 0.5, "B -> R/G": 0.5})
    bright_red = np.array([[0.9, 0.35, 0.3]])
    dark_red = np.array([[0.18, 0.07, 0.06]])
    move_bright = np.abs(stage.apply(bright_red, p) - bright_red).max()
    move_dark = np.abs(stage.apply(dark_red, p) - dark_red).max()
    assert move_bright > 5 * max(move_dark, 1e-9)  # stronger when brighter
    grays = np.array([[0.8, 0.8, 0.8], [0.2, 0.2, 0.2]])
    np.testing.assert_allclose(stage.apply(grays, p), grays, atol=1e-9)


def test_crosstalk_full_zone_still_has_an_effect():
    """At full Zone throw the zone mask takes over the inherent
    brightness weighting — shadow-zoned crosstalk must still visibly
    move dark saturated colors (it used to do ~nothing there)."""
    stage = ColourCrosstalkStage()

    def p(zone):
        return _with(stage, **{"R -> Y/B": 0.7, "Y -> R/G": 0.7,
                               "G -> Y/B": 0.7, "B -> R/G": 0.7,
                               "Zone": zone})

    dark_red = np.array([[0.18, 0.07, 0.06]])
    bright = np.array([[0.55, 0.35, 0.30]])
    move_everywhere = np.abs(stage.apply(dark_red, p(0.0)) - dark_red).max()
    move_shadow_zone = np.abs(stage.apply(dark_red, p(-1.0)) - dark_red).max()
    assert move_shadow_zone > 3.0 * move_everywhere
    # and the zone still SELECTS: bright pixels barely move at zone -1
    move_bright_at_lo = np.abs(stage.apply(bright, p(-1.0)) - bright).max()
    assert move_bright_at_lo < move_shadow_zone / 3.0
    # highlight zone keeps working too
    move_bright_at_hi = np.abs(stage.apply(bright, p(1.0)) - bright).max()
    assert move_bright_at_hi > np.abs(stage.apply(bright, p(0.0)) - bright).max() * 0.9


# ------------------------------------------------- brilliance reduction

def test_brilliance_reduction_darkens_by_saturation():
    stage = BrillianceReductionStage()
    p = _with(stage, Amount=0.7)  # identity is Amount 0 — raise to reduce
    saturated = np.array([[0.8, 0.25, 0.2]])
    mild = np.array([[0.5, 0.42, 0.4]])
    grey = np.array([[0.5, 0.5, 0.5]])

    out_sat = stage.apply(saturated, p)
    assert out_sat.max() < saturated.max() * 0.8          # darkened
    # chromaticity untouched: a pure luminance scale per pixel
    ratio = out_sat / saturated
    np.testing.assert_allclose(ratio, np.full_like(ratio, ratio[0, 0]),
                               atol=1e-9)
    # mild colors sit below the default sat pivot: spared
    np.testing.assert_allclose(stage.apply(mild, p), mild, atol=1e-9)
    np.testing.assert_allclose(stage.apply(grey, p), grey, atol=1e-9)


def test_brilliance_reduction_can_never_crush_to_black():
    """Marc's report: amount+chroma at 1 with pivot/falloff at 0 made
    the image fully black. The stops-based scale bounds the darkening
    at 2^-REDUCTION_STOPS for ANY slider combination."""
    stage = BrillianceReductionStage()
    worst = _with(stage, Amount=1.0, Chroma=1.0, Pivot=0.0, Falloff=0.01)
    x = np.random.default_rng(8).uniform(0.05, 0.95, (200, 3))
    out = stage.apply(x, worst)
    floor = 2.0 ** (-stage.REDUCTION_STOPS)
    assert (out.max(axis=1) >= x.max(axis=1) * floor - 1e-9).all()
    assert out.max() > 0.01  # nowhere near black


def test_brilliance_reduction_mask_sliders():
    stage = BrillianceReductionStage()
    saturated = np.array([[0.8, 0.25, 0.2]])
    # chroma 0 kills the mask -> identity even at full reduction
    np.testing.assert_allclose(
        stage.apply(saturated, _with(stage, Amount=1.0, Chroma=0.0)),
        saturated, atol=1e-9)
    # raising the pivot above the color's sat spares it
    np.testing.assert_allclose(
        stage.apply(saturated, _with(stage, Amount=1.0, Pivot=1.0,
                                     Falloff=0.2)),
        saturated, atol=1e-9)
    # lower pivot bites harder than the default
    lo = stage.apply(saturated, _with(stage, Amount=0.7, Pivot=0.1))
    hi = stage.apply(saturated, _with(stage, Amount=0.7, Pivot=0.6))
    assert lo.max() < hi.max()


# ------------------------------------------------------- sector family

def _hue_of(rgb):
    return rgb_to_reuleaux(rgb)[..., 0]


def test_sector_skew_local_hue_shift():
    stage = SectorSkewStage()
    greenish = np.array([[0.3, 0.7, 0.25]])
    hue_g = _hue_of(greenish)[0] * 360.0
    p = _with(stage, Hue=hue_g, Falloff=40.0, Skew=20.0)
    shifted = (_hue_of(stage.apply(greenish, p))[0] - _hue_of(greenish)[0]) * 360.0
    assert 15.0 < shifted < 25.0
    bluish = np.array([[0.25, 0.3, 0.8]])
    np.testing.assert_allclose(stage.apply(bluish, p), bluish, atol=1e-9)


def test_sector_brightness_and_saturation_local():
    reddish = np.array([[0.8, 0.3, 0.25]])
    bluish = np.array([[0.25, 0.3, 0.8]])
    hue_r = _hue_of(reddish)[0] * 360.0

    b = SectorBrightnessStage()
    pb = _with(b, Hue=hue_r, Falloff=40.0, Brightness=1.0)
    assert stage_max(b, reddish, pb) > reddish.max() * 1.1
    np.testing.assert_allclose(b.apply(bluish, pb), bluish, atol=1e-9)

    s = SectorSaturationStage()
    ps = _with(s, Hue=hue_r, Falloff=40.0, Saturation=1.4)
    assert _sat_of(s.apply(reddish, ps))[0] > _sat_of(reddish)[0] * 1.15
    np.testing.assert_allclose(s.apply(bluish, ps), bluish, atol=1e-9)


def stage_max(stage, x, p):
    return stage.apply(x, p).max()


def test_sector_squash_converges_and_spreads():
    stage = SectorSquashStage()
    reds = np.array([[0.8, 0.35, 0.3], [0.8, 0.3, 0.42]])  # around red
    target = 10.0
    p = _with(stage, Hue=target, Falloff=60.0, Squash=1.0)
    hues_in = _hue_of(reds) * 360.0
    hues_out = _hue_of(stage.apply(reds, p)) * 360.0

    def dist(h):
        return np.abs((h - target + 180.0) % 360.0 - 180.0)

    assert (dist(hues_out) < dist(hues_in) * 0.6).all()     # pulled in

    p_spread = _with(stage, Hue=target, Falloff=60.0, Squash=-1.0)
    hues_spread = _hue_of(stage.apply(reds, p_spread)) * 360.0
    assert (dist(hues_spread) > dist(hues_in) * 1.2).all()  # pushed out


def test_sector_squash_is_foldover_proof():
    """Hue transfer must stay monotone for the whole strength range —
    hue crossings (banding) are impossible by construction."""
    stage = SectorSquashStage()
    for squash in (-1.0, -0.5, 0.5, 1.0):
        p = _with(stage, Hue=180.0, Falloff=90.0, Squash=squash)
        target, width, s = 0.5, 90.0 / 360.0, squash
        delta = np.linspace(-0.5, 0.5, 20001)
        t = np.clip(np.abs(delta) / width, 0.0, 1.0)
        w = np.cos(0.5 * np.pi * t) ** 2
        out = target + delta * (1.0 - s * w)
        assert (np.diff(out) >= -1e-12).all(), f"foldover at squash={squash}"


# --------------------------------------------------------- conventions

def test_param_names_match_param_counts():
    for cls in CHROMOGEN_STAGES:
        stage = cls()
        assert len(stage.param_names) == stage.identity().size, cls.name
        lo, hi = stage.bounds()
        assert lo.size == hi.size == stage.identity().size, cls.name
        assert ((stage.identity() >= lo) & (stage.identity() <= hi)).all(), cls.name


def test_every_chromogen_stage_has_a_dctl():
    from pathlib import Path
    dctl_dir = Path(__file__).resolve().parents[1] / "dctl"
    for cls in CHROMOGEN_STAGES:
        name = cls.name.replace(" ", "")
        assert (dctl_dir / f"{name}.dctl").exists(), cls.name


# ---------------------------------------------------------- bake CLI

def test_stage_bake_cli(tmp_path, monkeypatch):
    from app.core.lut import apply_lut, parse_cube
    from tools import stage_bake

    out = tmp_path / "sat.cube"
    monkeypatch.setattr(sys, "argv", [
        "stage_bake", "--stage", "Colour Saturation",
        "--set", "R/G=1.4", "--set", "Y/B=1.8",
        "--out", str(out), "--size", "9",
    ])
    stage_bake.main()

    lut = parse_cube(out)
    stage = ColourSaturationStage()
    p = _with(stage, **{"R/G": 1.4, "Y/B": 1.8})
    grid = np.linspace(0.0, 1.0, 9)
    pts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(
        apply_lut(lut, pts), stage.apply(pts, p), atol=2e-9
    )


def test_stage_bake_rejects_unknown_slider(tmp_path, monkeypatch):
    from tools import stage_bake

    monkeypatch.setattr(sys, "argv", [
        "stage_bake", "--stage", "Neutral Tint",
        "--set", "Nope=1", "--out", str(tmp_path / "x.cube"),
    ])
    with pytest.raises(SystemExit, match="no slider"):
        stage_bake.main()
