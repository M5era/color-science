"""ME_Filmic Contrast port: identity, the DCTL's behavioural contract
(mid-grey pivot, end-exposure in stops, shoulder/toe rolls), the
extended Black Point range, and neutral-safety (the grey-locked tone
architecture depends on grey-in -> grey-out)."""

import numpy as np

from app.core.filmic import (
    MID_GREY,
    FilmicContrastStage,
    lin_to_logc3,
    logc3_to_lin,
)


def _with(stage, **by_name):
    p = stage.identity().copy()
    for name, value in by_name.items():
        p[stage.param_names.index(name)] = value
    return p


def _source(n=300, seed=9):
    x = np.random.default_rng(seed).uniform(0.0, 1.05, (n, 3))
    extra = np.array([[MID_GREY] * 3, [0.0] * 3, [1.4, 1.3, 1.2]])
    return np.concatenate([x, extra])


def test_identity_is_exact():
    s = FilmicContrastStage()
    x = _source()
    np.testing.assert_allclose(s.apply(x, s.identity()), x, atol=1e-12)


def test_init_is_engaged_but_gentle():
    """init() must differ from identity (gradient for the solver) while
    staying a subtle grade — the DCTL's own default shoulder. Ceiling
    0.15: the LIVE toe/shoulder position mappings (2026-07-22)
    strengthened the default rolls to ~0.11 at the extremes."""
    s = FilmicContrastStage()
    x = _source()
    out = s.apply(x, s.init())
    delta = np.abs(out - x).max()
    assert 1e-4 < delta < 0.15


def test_exposure_is_stops_and_achromatic():
    """+N stops moves every pixel's LINEAR value by exactly 2^N —
    mid-grey lands N stops up and chromaticity ratios are untouched."""
    s = FilmicContrastStage()
    x = np.array([[MID_GREY] * 3, [0.55, 0.38, 0.30]])
    out = s.apply(x, _with(s, Exposure=1.5))
    ratio = logc3_to_lin(out) / logc3_to_lin(x)
    np.testing.assert_allclose(ratio, 2.0 ** 1.5, rtol=1e-9)


def test_contrast_pivots_at_mid_grey():
    s = FilmicContrastStage()
    g = np.array([[MID_GREY] * 3])
    for c in (0.6, 1.5, 2.5):
        out = s.apply(g, _with(s, Contrast=c))
        np.testing.assert_allclose(out, g, atol=1e-8)
    # and Pivot moves the fixed point off mid-grey
    shifted = MID_GREY + 0.1
    out = s.apply(np.array([[shifted] * 3]), _with(s, Contrast=2.0, Pivot=0.1))
    np.testing.assert_allclose(out, [[shifted] * 3], atol=1e-8)


def test_contrast_steepens_around_pivot():
    s = FilmicContrastStage()
    lo = np.array([[MID_GREY - 0.1] * 3])
    hi = np.array([[MID_GREY + 0.1] * 3])
    p = _with(s, Contrast=1.6)
    assert s.apply(lo, p)[0, 0] < lo[0, 0]
    assert s.apply(hi, p)[0, 0] > hi[0, 0]


def test_white_point_rolls_highlights_only():
    s = FilmicContrastStage()
    p = _with(s, **{"White Point": 1.0})
    x = np.array([[0.3] * 3, [0.95] * 3])
    out = s.apply(x, p)
    np.testing.assert_allclose(out[0], x[0], atol=1e-12)  # below the pivot
    assert out[1, 0] < x[1, 0]                            # top compressed


def test_black_point_extended_range_keeps_lifting():
    """Marc 2026-07-22: range extended to 1.5. The stock sanitize floor
    (0.69) dead-zoned the slider past ~0.775; with the lowered floor the
    lift must keep growing monotonically to the end of the range."""
    s = FilmicContrastStage()
    black = np.array([[0.0, 0.0, 0.0]])
    lifts = [s.apply(black, _with(s, **{"Black Point": bp}))[0, 0]
             for bp in (0.5, 0.775, 1.0, 1.25, 1.5)]
    assert all(b > a + 1e-3 for a, b in zip(lifts, lifts[1:])), lifts
    assert lifts[-1] > 0.25   # 1.5 reaches a strong fade


def test_neutral_in_neutral_out():
    """Grey must stay grey for ANY parameters — the grey-locked tone
    freeze relies on the tone node being neutral-safe."""
    s = FilmicContrastStage()
    lo, hi = s.bounds()
    ramp = np.linspace(0.02, 1.0, 24)[:, None].repeat(3, axis=1)
    rng = np.random.default_rng(3)
    for _ in range(40):
        p = lo + (hi - lo) * rng.uniform(0.0, 1.0, lo.size)
        out = s.apply(ramp, p)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, out[:, :1].repeat(3, axis=1),
                                   atol=1e-9)


def test_preserve_color_tempers_saturation_gain():
    """Per-RGB contrast (Preserve Color 0) pushes saturation harder than
    the luma-only path (Preserve Color 1)."""
    s = FilmicContrastStage()
    x = np.array([[0.50, 0.40, 0.35]])

    def spread(p):
        out = s.apply(x, p)
        return float(out.max() - out.min())

    rich = spread(_with(s, Contrast=1.8, **{"Preserve Color": 0.0}))
    tame = spread(_with(s, Contrast=1.8, **{"Preserve Color": 1.0}))
    assert rich > tame


def test_white_point_extended_down_keeps_pulling():
    """Marc 2026-07-22: extended range going down — the stock floors
    dead-zoned the slider below ~0.42; the ceiling must now keep
    dropping monotonically toward mid-grey."""
    s = FilmicContrastStage()
    top = np.array([[1.0, 1.0, 1.0]])
    ceilings = [s.apply(top, _with(s, **{"White Point": wp}))[0, 0]
                for wp in (0.8, 0.42, 0.2, 0.0, -0.15)]
    assert all(b < a - 1e-3 for a, b in zip(ceilings, ceilings[1:])), ceilings
    assert ceilings[-1] < 0.55            # a real fade at the bottom


def test_shoulder_shapes_the_roll_when_wp_engaged():
    s = FilmicContrastStage()
    x = np.array([[0.7, 0.7, 0.7]])
    early = s.apply(x, _with(s, **{"White Point": 0.6, "Shoulder": 0.45}))
    late = s.apply(x, _with(s, **{"White Point": 0.6, "Shoulder": 0.95}))
    # roll starting below 0.7 compresses it; starting above leaves it
    assert late[0, 0] > early[0, 0] + 1e-3


def test_toe_falloff_widened_range_changes_shape():
    """The stock preserve-midgray squashed the whole falloff slider into
    strength 1.48..2.95; the remap (0.25..3.35) must give visibly
    different toes at the extremes while keeping the default anchored."""
    s = FilmicContrastStage()
    x = np.array([[0.12, 0.12, 0.12]])
    bp = {"Black Point": 0.8}
    sharp = s.apply(x, _with(s, **bp, **{"Toe Falloff": 0.0}))
    soft = s.apply(x, _with(s, **bp, **{"Toe Falloff": 10.0}))
    assert abs(sharp[0, 0] - soft[0, 0]) > 5e-3


def test_labels():
    s = FilmicContrastStage()
    assert s.label(s.identity()) == "(idle)"
    assert "punch" in s.label(_with(s, Contrast=1.5))
    assert len(s.short_label(_with(s, Contrast=1.5))) <= 9


import pytest


@pytest.mark.skip(reason="Python stage slimmed to 13 params on "
                  "2026-07-23 (Marc); DCTL work is deliberately "
                  "paused, so dctl/FilmicContrast.dctl still carries "
                  "the 20-slider layout. Re-enable (and update the "
                  "expected order) when the DCTL is slimmed to match.")
def test_dctl_slider_order_matches_param_names():
    """The float sliders in dctl/FilmicContrast.dctl must line up 1:1
    with param_names — the .drx patch relies on it."""
    import re
    from pathlib import Path
    text = Path(__file__).resolve().parents[1].joinpath(
        "dctl", "FilmicContrast.dctl").read_text()
    sliders = [m.group(1) for line in text.splitlines()
               if not line.lstrip().startswith("//")
               for m in [re.search(
                   r"DEFINE_UI_PARAMS\(\s*(\w+)[^)]*?DCTLUI_SLIDER_FLOAT",
                   line)] if m]
    assert len(sliders) == len(FilmicContrastStage.param_names)


def test_bend_point_is_stops_and_live_across_travel():
    """Bend Point reads in STOPS above mid grey (2026-07-23): the old
    code-linear mapping was dead until ~0.5 (only invisible speculars
    moved). Now every step of travel must visibly deepen the bend, and
    the top of the range is exact identity."""
    s = FilmicContrastStage()
    x = np.array([[0.9] * 3])
    top = s.apply(x, _with(s, **{"Bend Point": 8.5}))
    np.testing.assert_allclose(top, x, atol=1e-12)      # off at the top
    outs = [s.apply(x, _with(s, **{"Bend Point": bp}))[0, 0]
            for bp in (8.0, 6.5, 5.0, 3.5, 2.0)]
    assert all(b < a - 1e-3 for a, b in zip(outs, outs[1:])), outs


def test_shoulder_falloff_negative_is_geometric_and_alive():
    """Synced with the DCTL: below 0 the knee strength DOUBLES every 5
    slider units — every negative step must keep visibly tightening
    (the old linear mapping dead-zoned the bulk of the slider)."""
    from app.core.filmic import shoulder_strength
    assert shoulder_strength(0.0) == 10.0
    assert shoulder_strength(-5.0) == 20.0
    assert shoulder_strength(-25.0) == 320.0

    s = FilmicContrastStage()
    ramp = np.linspace(0.0, 1.0, 512)[:, None].repeat(3, axis=1)
    p = {"White Point": 0.6, "Shoulder": 0.45}
    outs = [s.apply(ramp, _with(s, **p, **{"Shoulder Falloff": f}))
            for f in (0.0, -5.0, -10.0, -15.0, -20.0, -25.0)]
    # each geometric step still reshapes the curve measurably (the old
    # linear mapping was already converged well before -20)
    steps = [np.abs(b - a).max() for a, b in zip(outs, outs[1:])]
    assert all(d > 3e-4 for d in steps), steps


def test_second_shoulder_stage_parked_then_shapes():
    """WP2 = 1.02 is exact identity; engaging it adds a second knee on
    top of the first roll."""
    s = FilmicContrastStage()
    x = _source()
    base = _with(s, **{"White Point": 0.9, "Shoulder": 0.5})
    one = s.apply(x, base)
    two = s.apply(x, _with(s, **{"White Point": 0.9, "Shoulder": 0.5,
                                 "Bend Point": 0.6,
                                 "Bend": 0.9,
                                 "Bend Falloff": -10.0}))
    assert not np.allclose(one, two)
    # below both roll pivots nothing moves
    lo = np.array([[0.2] * 3])
    np.testing.assert_allclose(
        s.apply(lo, _with(s, **{"Bend Point": 0.6})), lo, atol=1e-12)


