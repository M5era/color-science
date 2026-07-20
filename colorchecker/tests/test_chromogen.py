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
    ColourCrosstalkStage,
    ColourSaturationStage,
    ContrastBoostStage,
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


# ------------------------------------------------------ contrast boost

def test_contrast_boost_steepens_mids_rolls_highlights():
    stage = ContrastBoostStage()
    p = _with(stage, **{"Contrast Boost": 0.8})
    ramp = np.linspace(0.05, 2.5, 800)[:, None].repeat(3, axis=1)
    out = stage.apply(ramp, p)[:, 0]
    slope = np.gradient(out, ramp[:, 0])
    mid = np.abs(ramp[:, 0] - 0.4) < 0.1
    high = ramp[:, 0] > 2.0
    assert slope[mid].mean() > 1.5          # boosted midtones
    assert abs(slope[high].mean() - 1.0) < 0.1  # rolled back to slope 1


def test_contrast_boost_chroma_modes():
    stage = ContrastBoostStage()
    x = np.array([[0.55, 0.35, 0.25]])
    keep = stage.apply(x, _with(stage, **{"Contrast Boost": 0.8, "Chroma": 0.0}))
    film = stage.apply(x, _with(stage, **{"Contrast Boost": 0.8, "Chroma": 1.0}))
    # chroma 0: chromaticity untouched (reuleaux sat constant)
    np.testing.assert_allclose(_sat_of(keep), _sat_of(x), atol=1e-6)
    # chroma 1: exactly the per-channel curve -> sat rises here
    curve = stage._curve(x, 0.8, MID_GREY, MID_GREY + 6.0 * STOP)
    np.testing.assert_allclose(film, curve, atol=1e-12)
    assert _sat_of(film)[0] > _sat_of(x)[0]


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

def test_neutral_tint_pivot_focuses_tonal_band_and_keeps_val():
    """v2 semantics: Amount 0..1, Pivot -1..+1 sweeps the focus bump
    from the darkest section to the brightest — no dead zones."""
    stage = NeutralTintStage()
    highs = np.array([[0.9, 0.9, 0.9]])
    lows = np.array([[0.12, 0.12, 0.12]])

    warm_high = _with(stage, Hue=40.0, Amount=0.5, Pivot=1.0)
    for x in (highs, lows):
        out = stage.apply(x, warm_high)
        # val (contrast) untouched by construction
        np.testing.assert_allclose(out.max(axis=1), x.max(axis=1), atol=1e-9)
    assert _sat_of(stage.apply(highs, warm_high))[0] > 0.05   # tinted
    assert _sat_of(stage.apply(lows, warm_high))[0] < 0.02    # spared

    # leftmost pivot MUST grab the darkest section (v1's dead zone bug)
    cold_low = _with(stage, Hue=220.0, Amount=0.5, Pivot=-1.0)
    assert _sat_of(stage.apply(lows, cold_low))[0] > 0.05
    assert _sat_of(stage.apply(highs, cold_low))[0] < 0.02


def test_neutral_tint_amount_is_dye_convergence_not_gain():
    """Full amount pulls a focused neutral all the way to the dye
    anchor (sat = TINT_MAX_SAT) — bounded, saturating response."""
    stage = NeutralTintStage()
    # grey exactly at the default focus center (pivot 0 -> MID_GREY)
    grey = np.full((1, 3), 0.391)

    sats = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = _with(stage, Hue=40.0, Amount=a)
        sats.append(_sat_of(stage.apply(grey, p))[0])
    assert sats[0] < 1e-12                       # 0 = nothing
    assert all(b > a for a, b in zip(sats, sats[1:]))  # monotone
    np.testing.assert_allclose(sats[-1], stage.TINT_MAX_SAT, atol=1e-6)
    # eased start: quarter throw is well under a quarter of the range
    assert sats[1] < 0.25 * sats[-1] * 0.8


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
