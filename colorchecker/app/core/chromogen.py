"""Chromogen-style stages: behavior replication of FilmLight's
look-development tools, in reuleaux space (see ROADMAP.md, "Chromogen-
style stage family"). Slider sets copy Chromogen's panels literally
(Marc's directive + screenshots).

The standard MODULATION block on (almost) every tool is exactly three
sliders:
  Zone   (signed): 0 = everywhere, + = highlights only, - = shadows only
  Pivot  : the luma mask's pivot point, 0 = mid-grey (offset in working
           units around MID_GREY; the falloff is a fixed smooth width)
  Chroma (signed): 0 = everything, + = only saturated colors,
           - = only neutrals/desaturated (Marc's intuitive convention)
Falloff appears as a fourth slider only where Chromogen exposes one
(Highlight Bleach, Neutral Tint ramps; the sector tools' hue falloff).

Shared geometric primitive: the OPPONENT view of the reuleaux chroma
plane. Yellow (60 deg) and Blue (240 deg) are exactly antipodal in
reuleaux, so the Y/B axis is native; the orthogonal axis (150/330 deg)
plays the green-magenta ("R/G") axis.

Every stage has a companion DCTL in dctl/ with sliders in EXACTLY the
units printed by describe() (hue-like params in degrees, the rest raw).
"""

import numpy as np

from app.core.reuleaux import _spow, reuleaux_to_rgb, rgb_to_reuleaux
from app.core.stage_base import Stage
from app.core.windows import ramp_window, wrapped_window

_TWO_PI = 2.0 * np.pi

# Stops calibration (Arri LogC3 EI800: 18% grey encodes to ~0.391 and
# one scene stop spans ~0.0741 code values in the log region). All
# Pivot sliders are in STOPS relative to mid-grey; ramp Falloffs are a
# width in stops. Future: transfer-function dropdown (see ROADMAP Plan
# C notes) — for now Marc shoots LogC3 only.
MID_GREY = 0.391
STOP = 0.0740774
LUMA_FALLOFF = 0.5      # fixed smooth width of the Zone luma mask (code values)
SAT_GATE_PIVOT = 0.25   # fixed internal shape of the Chroma gate
SAT_GATE_FALLOFF = 0.5

# opponent axes in the reuleaux chroma plane (turns)
YB_AXIS_TURNS = 60.0 / 360.0    # yellow (+) <-> blue (-): exactly antipodal
RG_AXIS_TURNS = 150.0 / 360.0   # green-cyan (+) <-> magenta-red (-)


# ------------------------------------------------------------ helpers

def modulation(val, sat, zone, pivot, chroma):
    """The standard Zone / Pivot / Chroma weight, in [0, 1].
    `pivot` is in STOPS from mid-grey. Identity (weight 1 everywhere)
    at zone == 0 and chroma == 0."""
    r = ramp_window(val, MID_GREY + pivot * STOP, LUMA_FALLOFF)
    m_luma = 1.0 - abs(zone) + abs(zone) * (r if zone >= 0.0 else 1.0 - r)
    rs = ramp_window(sat, SAT_GATE_PIVOT, SAT_GATE_FALLOFF)
    m_chroma = 1.0 - abs(chroma) + abs(chroma) * (
        rs if chroma >= 0.0 else 1.0 - rs
    )
    return m_luma * m_chroma


def to_chroma_vec(hue, sat):
    ang = hue * _TWO_PI
    return sat * np.cos(ang), sat * np.sin(ang)


def from_chroma_vec(c1, c2):
    sat = np.hypot(c1, c2)
    hue = np.arctan2(c2, c1) / _TWO_PI
    return hue % 1.0, sat


def _axis_dir(turns):
    ang = turns * _TWO_PI
    return np.cos(ang), np.sin(ang)


# 4-anchor (R, Y, G, B) piecewise-linear hue weighting with wrap,
# mirroring the validated Broad stage's 9-point interp trick
_RYGB_XS = np.array([2.0 / 3.0 - 1.0, 0.0, 1.0 / 6.0, 1.0 / 3.0,
                     2.0 / 3.0, 1.0, 1.0 / 6.0 + 1.0])
_RYGB_WRAP = [3, 0, 1, 2, 3, 0, 1]  # B, R, Y, G, B, R, Y


def rygb_interp(hue, amounts):
    """amounts: 4 values for R, Y, G, B -> smooth-ish per-hue amount."""
    ys = np.asarray(amounts, dtype=np.float64)[_RYGB_WRAP]
    return np.interp(hue, _RYGB_XS, ys)


def _softplus(x, width):
    return width * np.logaddexp(0.0, x / width)


# ------------------------------------------------------------- stages

class ColourSaturationStage(Stage):
    """Chromogen Colour Saturation: scale the two opponent axes of the
    chroma plane independently (R/G = green-magenta, Y/B = yellow-blue;
    keep them equal for the ganged move), plus the standard modulation.
    Reconstructs at constant val — anisotropic scaling bends hues
    toward the stronger axis, as the real tool visibly does."""

    name = "Colour Saturation"
    param_names = ["R/G", "Y/B", "Zone", "Pivot", "Chroma"]

    def identity(self):
        return np.array([1.0, 1.0, 0.0, 0.0, 0.0])

    def bounds(self):
        lo = [0.0, 0.0, -1.0, -6.0, -1.0]
        hi = [3.0, 3.0, 1.0, 8.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        s_rg, s_yb, zone, pivot, chroma = params
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        m = modulation(val, sat, zone, pivot, chroma)
        c1, c2 = to_chroma_vec(hue, sat)

        theta = YB_AXIS_TURNS * _TWO_PI
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        u = cos_t * c1 + sin_t * c2      # along Y/B
        v = -sin_t * c1 + cos_t * c2     # along R/G (green-magenta)

        u = u * (1.0 + m * (s_yb - 1.0))
        v = v * (1.0 + m * (s_rg - 1.0))

        c1 = cos_t * u - sin_t * v
        c2 = sin_t * u + cos_t * v
        hue2, sat2 = from_chroma_vec(c1, c2)
        return reuleaux_to_rgb(np.stack([hue2, sat2, val], axis=-1))

    def describe(self, params):
        s_rg, s_yb, zone, pivot, chroma = params
        return "\n".join([
            "Colour Saturation (paste into dctl/ColourSaturation.dctl):",
            f"  R/G {s_rg:.3f}   Y/B {s_yb:.3f}",
            f"  Zone {zone:+.3f}  Pivot {pivot:+.3f}  Chroma {chroma:+.3f}",
        ])


class ColourCrosstalkStage(Stage):
    """Chromogen Colour Crosstalk (the 'twister'): each R/Y/G/B sector's
    chroma is displaced along the ORTHOGONAL opponent axis (R,G -> along
    Y/B; Y,B -> along R/G). The luminance dependence is INHERENT — the
    displacement scales with brightness (val), so the whole space tilts:
    shadows barely move, highlights move fully. It also scales with sat,
    so neutrals never move. Zone/Pivot/Chroma modulate on top."""

    name = "Colour Crosstalk"
    param_names = ["R -> Y/B", "Y -> R/G", "G -> Y/B", "B -> R/G",
                   "Zone", "Pivot", "Chroma"]

    # sector index -> displacement axis (R, G along Y/B; Y, B along R/G)
    _AXES = (YB_AXIS_TURNS, RG_AXIS_TURNS, YB_AXIS_TURNS, RG_AXIS_TURNS)

    def identity(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def bounds(self):
        lo = [-1.0, -1.0, -1.0, -1.0, -1.0, -6.0, -1.0]
        hi = [1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        amounts, zone, pivot, chroma = params[:4], params[4], params[5], params[6]
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        m = modulation(val, sat, zone, pivot, chroma)
        c1, c2 = to_chroma_vec(hue, sat)
        for i in range(4):
            one_hot = np.zeros(4)
            one_hot[i] = 1.0
            w = rygb_interp(hue, one_hot) * sat * val * m * amounts[i]
            d1, d2 = _axis_dir(self._AXES[i])
            c1 = c1 + w * d1
            c2 = c2 + w * d2
        hue2, sat2 = from_chroma_vec(c1, c2)
        return reuleaux_to_rgb(np.stack([hue2, sat2, val], axis=-1))

    def describe(self, params):
        r, y, g, b, zone, pivot, chroma = params
        return "\n".join([
            "Colour Crosstalk (paste into dctl/ColourCrosstalk.dctl):",
            f"  R -> Y/B {r:+.3f}  Y -> R/G {y:+.3f}  "
            f"G -> Y/B {g:+.3f}  B -> R/G {b:+.3f}",
            f"  Zone {zone:+.3f}  Pivot {pivot:+.3f}  Chroma {chroma:+.3f}",
        ])


class ContrastBoostStage(Stage):
    """Chromogen Contrast Boost: analytic smooth contrast — slope
    (1 + boost) around the grey pivot, smoothly returning to slope 1
    above the highlight pivot (filmic shoulder; push Highlight Pivot to
    max to disable it). Both pivots are in STOPS from mid-grey. The
    Chroma mix blends val-only application (0 = chromaticity untouched)
    with per-RGB-channel (1 = sat rises with contrast); default 0.5."""

    name = "Contrast Boost"
    param_names = ["Contrast Boost", "Grey Pivot", "Highlight Pivot", "Chroma"]

    _SHOULDER = 0.15  # softplus width of the highlight return, val units

    def identity(self):
        return np.array([0.0, 0.0, 6.0, 0.5])

    def bounds(self):
        return (np.asarray([-0.9, -4.0, 0.5, 0.0]),
                np.asarray([2.0, 4.0, 14.0, 1.0]))

    def _curve(self, v, boost, grey_abs, highlight_abs):
        w = self._SHOULDER
        return v + boost * (
            (v - grey_abs)
            - _softplus(v - highlight_abs, w)
            + _softplus(grey_abs - highlight_abs, w)
        )

    def apply(self, x, params):
        boost, grey_pivot, highlight_pivot, mix = params
        grey_abs = MID_GREY + grey_pivot * STOP
        highlight_abs = MID_GREY + highlight_pivot * STOP
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        val_mode = reuleaux_to_rgb(np.stack(
            [hue, sat, self._curve(val, boost, grey_abs, highlight_abs)],
            axis=-1,
        ))
        rgb_mode = self._curve(x, boost, grey_abs, highlight_abs)
        return (1.0 - mix) * val_mode + mix * rgb_mode

    def describe(self, params):
        boost, grey, highlight, mix = params
        return "\n".join([
            "Contrast Boost (paste into dctl/ContrastBoost.dctl):",
            f"  Contrast Boost {boost:+.3f}   Grey Pivot {grey:+.3f}   "
            f"Highlight Pivot {highlight:+.3f}   Chroma {mix:.3f}",
        ])


class HighlightBleachStage(Stage):
    """Chromogen Highlight Bleach: per-sector (R/Y/G/B) desaturation of
    the highlights at constant val — amounts x a highlight ramp (Pivot
    offset from mid-grey + Falloff, smooth early kick-in) x the Chroma
    gate. Unganged sectors are the demo's save-the-blue-skies /
    keep-the-skin-yellows moves."""

    name = "Highlight Bleach"
    param_names = ["R", "Y", "G", "B", "Pivot", "Falloff", "Chroma"]

    def identity(self):
        return np.array([0.0, 0.0, 0.0, 0.0, -2.0, 4.0, 0.0])

    def bounds(self):
        lo = [0.0, 0.0, 0.0, 0.0, -6.0, 0.5, -1.0]
        hi = [1.0, 1.0, 1.0, 1.0, 8.0, 16.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        amounts, pivot, falloff, chroma = (
            params[:4], params[4], params[5], params[6]
        )
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        w = (
            rygb_interp(hue, amounts)
            * ramp_window(val, MID_GREY + pivot * STOP, falloff * STOP)
            * modulation(val, sat, 0.0, 0.0, chroma)
        )
        sat2 = sat * (1.0 - w)
        return reuleaux_to_rgb(np.stack([hue, sat2, val], axis=-1))

    def describe(self, params):
        r, y, g, b, pivot, falloff, chroma = params
        return "\n".join([
            "Highlight Bleach (paste into dctl/HighlightBleach.dctl):",
            f"  R {r:.3f}  Y {y:.3f}  G {g:.3f}  B {b:.3f}",
            f"  Pivot {pivot:+.3f}  Falloff {falloff:.3f}  Chroma {chroma:+.3f}",
        ])


class NeutralTintStage(Stage):
    """Chromogen Neutral Tint: tint toward a picked hue with a SIGNED
    amount (+ = highlights, - = shadows) at constant val — contrast
    untouched by construction. Pivot/Falloff are in stops; the
    Chroma gate defaults protective (only-neutrals) so saturated colors
    aren't pushed toward a gamut edge. Two instances = warm highs +
    cold lows."""

    name = "Neutral Tint"
    param_names = ["Hue", "Amount", "Pivot", "Falloff", "Chroma"]

    # slider +-1.0 maps to +-TINT_SCALE in reuleaux sat units — a raw
    # sat push of 0.5 was WAY too aggressive (Marc); mid-slider should
    # be a subtle tint, full throw strong but usable
    TINT_SCALE = 0.25

    def identity(self):
        return np.array([0.0, 0.0, 0.0, 4.0, -0.5])

    def bounds(self):
        lo = [0.0, -1.0, -6.0, 0.5, -1.0]
        hi = [360.0, 1.0, 8.0, 16.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        hue_deg, amount, pivot, falloff, chroma = params
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        r = ramp_window(val, MID_GREY + pivot * STOP, falloff * STOP)
        side = r if amount >= 0.0 else 1.0 - r
        m = side * modulation(val, sat, 0.0, 0.0, chroma)

        strength = abs(amount) * self.TINT_SCALE
        d1, d2 = _axis_dir(hue_deg / 360.0)
        c1, c2 = to_chroma_vec(hue, sat)
        c1 = c1 + strength * m * d1
        c2 = c2 + strength * m * d2
        hue2, sat2 = from_chroma_vec(c1, c2)
        return reuleaux_to_rgb(np.stack([hue2, sat2, val], axis=-1))

    def describe(self, params):
        hue_deg, amount, pivot, falloff, chroma = params
        where = "highlights" if amount >= 0 else "shadows"
        return "\n".join([
            "Neutral Tint (paste into dctl/NeutralTint.dctl):",
            f"  Hue {hue_deg:.1f}°  Amount {amount:+.3f} (tints {where})",
            f"  Pivot {pivot:+.3f}  Falloff {falloff:.3f}  Chroma {chroma:+.3f}",
        ])


# ------------------------------------------------------ sector family

class _SectorStage(Stage):
    """Shared machinery for the single-picked-hue sector tools: a
    wrapped cos^2 hue window (Hue + Falloff, degrees) times the
    standard Zone/Pivot/Chroma modulation gates one adjustment.
    Param vector: [hue, amount, falloff, zone, pivot, chroma]."""

    # subclasses override index 1 with their tool's slider name
    param_names = ["Hue", "Amount", "Falloff", "Zone", "Pivot", "Chroma"]

    _AMOUNT_ID = 0.0
    _AMOUNT_LO = -1.0
    _AMOUNT_HI = 1.0

    def identity(self):
        return np.array([0.0, self._AMOUNT_ID, 60.0, 0.0, 0.0, 0.0])

    def bounds(self):
        lo = [0.0, self._AMOUNT_LO, 5.0, -1.0, -6.0, -1.0]
        hi = [360.0, self._AMOUNT_HI, 180.0, 1.0, 8.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def _weight(self, hue, sat, val, params):
        center = (params[0] / 360.0) % 1.0
        return (
            wrapped_window(hue, center, 0.0, params[2] / 360.0)
            * modulation(val, sat, params[3], params[4], params[5])
        )

    def describe(self, params):
        center, amount, falloff, zone, pivot, chroma = params
        return "\n".join([
            f"{self.name} (paste into dctl/{self.name.replace(' ', '')}.dctl):",
            f"  Hue {center:.1f}°  {self.param_names[1]} {amount:+.3f}  "
            f"Falloff {falloff:.1f}°",
            f"  Zone {zone:+.3f}  Pivot {pivot:+.3f}  Chroma {chroma:+.3f}",
        ])


class SectorSkewStage(_SectorStage):
    """Shift the hues of one picked sector (Chromogen Sector Skew).
    Skew in degrees of hue shift at full window weight."""

    name = "Sector Skew"
    param_names = ["Hue", "Skew", "Falloff", "Zone", "Pivot", "Chroma"]
    _AMOUNT_LO, _AMOUNT_HI = -60.0, 60.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        w = self._weight(hue, sat, val, params)
        hue2 = hue + w * (params[1] / 360.0)
        return reuleaux_to_rgb(np.stack([hue2, sat, val], axis=-1))


class SectorBrightnessStage(_SectorStage):
    """Brighten/darken one picked sector (Chromogen Sector Brightness).
    Val effect scales with sat (reuleaux convention) so the neutral
    axis is untouched."""

    name = "Sector Brightness"
    param_names = ["Hue", "Brightness", "Falloff", "Zone", "Pivot", "Chroma"]
    _AMOUNT_LO, _AMOUNT_HI = -3.0, 3.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        w = self._weight(hue, sat, val, params)
        val2 = val * np.maximum(1.0 + sat * (w * params[1]), 1e-6)
        return reuleaux_to_rgb(np.stack([hue, sat, val2], axis=-1))


class SectorSaturationStage(_SectorStage):
    """Saturate/desaturate one picked sector (Chromogen Sector
    Saturation). Amount is the DCTL-style sat factor (1 = neutral)."""

    name = "Sector Saturation"
    param_names = ["Hue", "Saturation", "Falloff", "Zone", "Pivot", "Chroma"]
    _AMOUNT_ID = 1.0
    _AMOUNT_LO, _AMOUNT_HI = 0.05, 2.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        w = self._weight(hue, sat, val, params)
        sat2 = _spow(sat, 1.0 / (1.0 + w * (params[1] - 1.0)))
        return reuleaux_to_rgb(np.stack([hue, sat2, val], axis=-1))


class SectorSquashStage(_SectorStage):
    """Compress (or, negative, SPREAD) the hues of one sector toward /
    away from the picked hue (Chromogen Sector Squash):

        h' = T + delta * (1 - s_eff * w(delta))

    with w the cos^2 window over the falloff. For this window shape,
    s_eff in [-1, 1] is inherently foldover-proof: the hue transfer's
    slope stays >= 0 everywhere (equality only at full squash at the
    center), so hue crossings/banding are mathematically impossible.
    The modulation gates s (its Chroma gate keeps noisy near-neutral
    hues out — that's what makes squash safe here)."""

    name = "Sector Squash"
    param_names = ["Hue", "Squash", "Falloff", "Zone", "Pivot", "Chroma"]
    _AMOUNT_LO, _AMOUNT_HI = -1.0, 1.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        target = (params[0] / 360.0) % 1.0
        width = max(params[2] / 360.0, 1e-6)

        delta = ((hue - target + 0.5) % 1.0) - 0.5
        t = np.clip(np.abs(delta) / width, 0.0, 1.0)
        w = np.cos(0.5 * np.pi * t) ** 2

        m = modulation(val, sat, params[3], params[4], params[5])
        s_eff = params[1] * m
        hue2 = target + delta * (1.0 - s_eff * w)
        return reuleaux_to_rgb(np.stack([hue2, sat, val], axis=-1))


CHROMOGEN_STAGES = [
    ColourSaturationStage,
    ColourCrosstalkStage,
    ContrastBoostStage,
    HighlightBleachStage,
    NeutralTintStage,
    SectorSkewStage,
    SectorBrightnessStage,
    SectorSaturationStage,
    SectorSquashStage,
]
