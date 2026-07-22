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
    staying a subtle grade — the DCTL's own default shoulder."""
    s = FilmicContrastStage()
    x = _source()
    out = s.apply(x, s.init())
    delta = np.abs(out - x).max()
    assert 1e-4 < delta < 0.1


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


def test_pin_ends_returns_extremes_to_input():
    s = FilmicContrastStage()
    x = np.array([[0.03] * 3, [MID_GREY] * 3, [1.0] * 3])
    strong = _with(s, Contrast=2.2)
    pinned = _with(s, Contrast=2.2, **{"Pin Ends": 1.0})
    err_free = np.abs(s.apply(x, strong) - x)
    err_pin = np.abs(s.apply(x, pinned) - x)
    # the deep shadow end comes back toward the untouched input
    assert err_pin[0, 0] < err_free[0, 0]


def test_pop_mids_darkens_the_band_and_holds_mid_grey_exactly():
    """Marc 2026-07-22: 'compensate exposure to keep mid grey intact' —
    the toolkit adds a global counter-gain so Pop Mids NEVER moves a
    mid-grey pixel (stock drifted it via the band feather), fully
    decoupling Pop Mids from Exposure in the fit."""
    s = FilmicContrastStage()
    x = np.array([[0.25] * 3, [MID_GREY] * 3])
    for pop in (-0.8, 0.5, 1.0, 3.0):
        out = s.apply(x, _with(s, **{"Pop Mids": pop}))
        np.testing.assert_allclose(out[1], x[1], atol=1e-12,
                                   err_msg=f"pop {pop}")  # mid EXACT
    out = s.apply(x, _with(s, **{"Pop Mids": 1.0}))
    assert out[0, 0] < x[0, 0] - 1e-4     # in the band: bitten down
    # and it stays exact with the tone curve engaged (curved mid anchor)
    p = _with(s, Contrast=1.6, **{"Pop Mids": 2.0, "White Point": 0.8})
    base = s.apply(np.array([[MID_GREY] * 3]),
                   _with(s, Contrast=1.6, **{"White Point": 0.8}))
    popped = s.apply(np.array([[MID_GREY] * 3]), p)
    np.testing.assert_allclose(popped, base, atol=1e-12)


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


def test_flare_lifts_shadows():
    s = FilmicContrastStage()
    x = np.array([[0.05] * 3])
    up = s.apply(x, _with(s, Flare=1.0))
    down = s.apply(x, _with(s, Flare=-1.0))
    assert up[0, 0] > x[0, 0] > down[0, 0]


def test_labels():
    s = FilmicContrastStage()
    assert s.label(s.identity()) == "(idle)"
    assert "punch" in s.label(_with(s, Contrast=1.5))
    assert len(s.short_label(_with(s, Contrast=1.5))) <= 9


def test_dctl_slider_order_matches_param_names():
    """The 14 float sliders in dctl/FilmicContrast.dctl must line up
    1:1 with param_names — the .drx patch relies on it."""
    import re
    from pathlib import Path
    text = Path(__file__).resolve().parents[1].joinpath(
        "dctl", "FilmicContrast.dctl").read_text()
    sliders = [m.group(1) for line in text.splitlines()
               if not line.lstrip().startswith("//")
               for m in [re.search(
                   r"DEFINE_UI_PARAMS\(\s*(\w+)[^)]*?DCTLUI_SLIDER_FLOAT",
                   line)] if m]
    assert len(sliders) == len(FilmicContrastStage.param_names) == 14
    # spot-check the anchoring entries
    assert sliders[0] == "P_exposure"
    assert sliders[6] == "p_black_point"
    assert sliders[13] == "p_flare"
