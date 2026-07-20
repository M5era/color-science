"""Chromogen-style stages: behavior replication of FilmLight's
look-development tools, in reuleaux space (see ROADMAP.md, "Chromogen-
style stage family").

All stages obey the standard contract (flat params, box bounds,
identity anchor, pure vectorized apply) and share two primitives:

- the MODULATION block (Zone / Pivot / Falloff / Chroma): every tool's
  effect weight. Zone is signed (+ = highlights, - = shadows, 0 =
  everywhere); Chroma is signed (+ = only already-saturated colors,
  - = only desaturated colors, 0 = uniform).
- the OPPONENT view of the reuleaux chroma plane: hue/sat as a 2-D
  chroma vector. Yellow (60 deg) and Blue (240 deg) are exactly
  antipodal in reuleaux, so the Y/B axis is native; the orthogonal
  axis (150/330 deg) plays the green-magenta ("R/G") axis.

Every stage has a companion DCTL in dctl/ with sliders in EXACTLY the
units printed by describe() (hue-like params in degrees, the rest
raw). Chain order guidance and typical uses are in the roadmap.
"""

import numpy as np

from app.core.reuleaux import _spow, reuleaux_to_rgb, rgb_to_reuleaux
from app.core.stage_base import Stage
from app.core.windows import ramp_window, wrapped_window

_TWO_PI = 2.0 * np.pi

# fixed internal shape of the Chroma modulation gate (sat axis)
SAT_GATE_PIVOT = 0.25
SAT_GATE_FALLOFF = 0.5

# opponent axes in the reuleaux chroma plane (turns)
YB_AXIS_TURNS = 60.0 / 360.0    # yellow (+) <-> blue (-): exactly antipodal
RG_AXIS_TURNS = 150.0 / 360.0   # green-cyan (+) <-> magenta-red (-)


# ------------------------------------------------------------ helpers

def modulation(val, sat, zone, pivot, falloff, chroma):
    """The shared Zone/Pivot/Falloff/Chroma effect weight, in [0, 1].
    Identity (weight 1 everywhere) at zone == 0 and chroma == 0."""
    r = ramp_window(val, pivot, falloff)
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
_RYGB = np.array([0.0, 1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0])
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
    chroma plane independently (R/G = green-magenta, Y/B = yellow-blue,
    ganged when equal), with an axis Rotate and the modulation block.
    Reconstructs at constant val — anisotropic scaling bends hues
    toward the stronger axis, as the real tool visibly does."""

    name = "Colour Saturation"
    param_names = ["RG Saturation", "YB Saturation", "Rotate",
                   "Zone", "Pivot", "Falloff", "Chroma"]

    def identity(self):
        return np.array([1.0, 1.0, 0.0, 0.0, 0.4, 0.5, 0.0])

    def bounds(self):
        lo = [0.0, 0.0, -45.0, -1.0, -0.5, 0.02, -1.0]
        hi = [3.0, 3.0, 45.0, 1.0, 2.0, 2.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        s_rg, s_yb, rotate, zone, pivot, falloff, chroma = params
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        m = modulation(val, sat, zone, pivot, falloff, chroma)
        c1, c2 = to_chroma_vec(hue, sat)

        theta = (YB_AXIS_TURNS + rotate / 360.0) * _TWO_PI
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
        s_rg, s_yb, rotate, zone, pivot, falloff, chroma = params
        return "\n".join([
            "Colour Saturation (paste into dctl/ColourSaturation.dctl):",
            f"  RG Saturation {s_rg:.3f}   YB Saturation {s_yb:.3f}   "
            f"Rotate {rotate:+.1f}°",
            f"  Zone {zone:+.3f}  Pivot {pivot:.3f}  Falloff {falloff:.3f}  "
            f"Chroma {chroma:+.3f}",
        ])


class ContrastBoostStage(Stage):
    """Chromogen Contrast Boost: analytic smooth contrast — slope
    (1 + boost) around the grey pivot, smoothly returning to slope 1
    above the highlight pivot (the HDR-safe rolloff). The Chroma mix
    blends val-only application (0 = chromaticity untouched, 'base
    grade') with per-RGB-channel (1 = sat rises with contrast, 'film
    grade'); Chromogen's own default is 0.5."""

    name = "Contrast Boost"
    param_names = ["Boost", "Grey Pivot", "Highlight Pivot", "Chroma"]

    _SHOULDER = 0.15  # softplus width of the highlight return, val units

    def identity(self):
        return np.array([0.0, 0.4, 0.9, 0.5])

    def bounds(self):
        return (np.asarray([-0.9, 0.0, 0.3, 0.0]),
                np.asarray([2.0, 1.0, 3.0, 1.0]))

    def _curve(self, v, boost, pivot, highlight):
        w = self._SHOULDER
        return v + boost * (
            (v - pivot)
            - _softplus(v - highlight, w)
            + _softplus(pivot - highlight, w)
        )

    def apply(self, x, params):
        boost, pivot, highlight, mix = params
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        val_mode = reuleaux_to_rgb(np.stack(
            [hue, sat, self._curve(val, boost, pivot, highlight)], axis=-1
        ))
        rgb_mode = self._curve(x, boost, pivot, highlight)
        return (1.0 - mix) * val_mode + mix * rgb_mode

    def describe(self, params):
        boost, pivot, highlight, mix = params
        return "\n".join([
            "Contrast Boost (paste into dctl/ContrastBoost.dctl):",
            f"  Boost {boost:+.3f}   Grey Pivot {pivot:.3f}   "
            f"Highlight Pivot {highlight:.3f}   Chroma {mix:.3f}",
        ])


class HighlightBleachStage(Stage):
    """Chromogen Highlight Bleach: per-sector (R/Y/G/B) desaturation of
    the highlights at constant val — amounts x a highlight ramp
    (pivot + falloff, smooth early kick-in) x the signed chroma gate.
    Unganged sectors are the demo's save-the-blue-skies /
    keep-the-skin-yellows moves."""

    name = "Highlight Bleach"
    param_names = ["Bleach R", "Bleach Y", "Bleach G", "Bleach B",
                   "Pivot", "Falloff", "Chroma"]

    def identity(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.0])

    def bounds(self):
        lo = [0.0, 0.0, 0.0, 0.0, -0.5, 0.02, -1.0]
        hi = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        amounts, pivot, falloff, chroma = params[:4], params[4], params[5], params[6]
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        w = (
            rygb_interp(hue, amounts)
            * ramp_window(val, pivot, falloff)
            * modulation(val, sat, 0.0, pivot, falloff, chroma)
        )
        sat2 = sat * (1.0 - w)
        return reuleaux_to_rgb(np.stack([hue, sat2, val], axis=-1))

    def describe(self, params):
        r, y, g, b, pivot, falloff, chroma = params
        return "\n".join([
            "Highlight Bleach (paste into dctl/HighlightBleach.dctl):",
            f"  Bleach R {r:.3f}  Y {y:.3f}  G {g:.3f}  B {b:.3f}",
            f"  Pivot {pivot:.3f}  Falloff {falloff:.3f}  Chroma {chroma:+.3f}",
        ])


class NeutralTintStage(Stage):
    """Chromogen Neutral Tint: tint toward a picked hue with a SIGNED
    amount (+ = highlights, - = shadows) at constant val — contrast
    untouched by construction. The chroma gate (default protective)
    fades the tint out for already-saturated colors so nothing gets
    pushed toward a gamut edge. Two instances = warm highs + cold lows."""

    name = "Neutral Tint"
    param_names = ["Hue", "Amount", "Pivot", "Falloff", "Chroma"]

    def identity(self):
        return np.array([0.0, 0.0, 0.4, 0.6, -0.5])

    def bounds(self):
        lo = [0.0, -0.5, -0.5, 0.02, -1.0]
        hi = [360.0, 0.5, 2.0, 2.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        hue_deg, amount, pivot, falloff, chroma = params
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        r = ramp_window(val, pivot, falloff)
        side = r if amount >= 0.0 else 1.0 - r
        m = side * modulation(val, sat, 0.0, pivot, falloff, chroma)

        d1, d2 = _axis_dir(hue_deg / 360.0)
        c1, c2 = to_chroma_vec(hue, sat)
        c1 = c1 + abs(amount) * m * d1
        c2 = c2 + abs(amount) * m * d2
        hue2, sat2 = from_chroma_vec(c1, c2)
        return reuleaux_to_rgb(np.stack([hue2, sat2, val], axis=-1))

    def describe(self, params):
        hue_deg, amount, pivot, falloff, chroma = params
        where = "highlights" if amount >= 0 else "shadows"
        return "\n".join([
            "Neutral Tint (paste into dctl/NeutralTint.dctl):",
            f"  Hue {hue_deg:.1f}°  Amount {amount:+.3f} (tints {where})",
            f"  Pivot {pivot:.3f}  Falloff {falloff:.3f}  Chroma {chroma:+.3f}",
        ])


class ColourCrosstalkStage(Stage):
    """Chromogen Colour Crosstalk (the 'twister'), approximated: each
    R/Y/G/B sector's chroma is displaced along the ORTHOGONAL opponent
    axis (R,G -> along Y/B; Y,B -> along R/G), with LUMINANCE AS THE
    WEIGHT — a monotone ramp, so shadows barely move and highlights
    move fully (tilt/shift of the whole space). Displacement scales
    with sat, so neutrals never move. Ganged equal amounts = the
    global twist; unganged = sector-local skews."""

    name = "Colour Crosstalk"
    param_names = ["Crosstalk R", "Crosstalk Y", "Crosstalk G",
                   "Crosstalk B", "Pivot", "Falloff"]

    def identity(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.8])

    def bounds(self):
        lo = [-0.5, -0.5, -0.5, -0.5, -0.5, 0.02]
        hi = [0.5, 0.5, 0.5, 0.5, 2.0, 2.0]
        return np.asarray(lo), np.asarray(hi)

    # sector index -> displacement axis (R, G along Y/B; Y, B along R/G)
    _AXES = (YB_AXIS_TURNS, RG_AXIS_TURNS, YB_AXIS_TURNS, RG_AXIS_TURNS)

    def apply(self, x, params):
        amounts, pivot, falloff = params[:4], params[4], params[5]
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        lum = ramp_window(val, pivot, falloff)
        c1, c2 = to_chroma_vec(hue, sat)
        for i in range(4):
            one_hot = np.zeros(4)
            one_hot[i] = 1.0
            w = rygb_interp(hue, one_hot) * lum * sat * amounts[i]
            d1, d2 = _axis_dir(self._AXES[i])
            c1 = c1 + w * d1
            c2 = c2 + w * d2
        hue2, sat2 = from_chroma_vec(c1, c2)
        return reuleaux_to_rgb(np.stack([hue2, sat2, val], axis=-1))

    def describe(self, params):
        r, y, g, b, pivot, falloff = params
        return "\n".join([
            "Colour Crosstalk (paste into dctl/ColourCrosstalk.dctl):",
            f"  Crosstalk R {r:+.3f}  Y {y:+.3f}  G {g:+.3f}  B {b:+.3f}",
            f"  Pivot {pivot:.3f}  Falloff {falloff:.3f}",
        ])


# ------------------------------------------------------ sector family

class _SectorStage(Stage):
    """Shared machinery for the single-picked-hue sector tools:
    a wrapped cos^2 hue window (center + falloff) times the modulation
    block gates one adjustment. Params [0..1] = hue center (deg),
    falloff (deg); [2] = the tool's amount; [3..6] = modulation."""

    # subclasses override with their tool's slider name at index 2,
    # matching their companion DCTL exactly
    param_names = ["Hue", "Falloff", "Amount",
                   "Zone", "Pivot", "Luma Falloff", "Chroma"]

    _AMOUNT_ID = 0.0
    _AMOUNT_LO = -1.0
    _AMOUNT_HI = 1.0

    def identity(self):
        return np.array([0.0, 60.0, self._AMOUNT_ID, 0.0, 0.4, 0.6, 0.0])

    def bounds(self):
        lo = [0.0, 5.0, self._AMOUNT_LO, -1.0, -0.5, 0.02, -1.0]
        hi = [360.0, 180.0, self._AMOUNT_HI, 1.0, 2.0, 2.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def _weight(self, hue, sat, val, params):
        center, falloff = params[0] / 360.0, params[1] / 360.0
        return (
            wrapped_window(hue, center % 1.0, 0.0, falloff)
            * modulation(val, sat, params[3], params[4], params[5], params[6])
        )

    def describe(self, params):
        center, falloff, amount, zone, pivot, lfall, chroma = params
        return "\n".join([
            f"{self.name} (paste into dctl/{self.name.replace(' ', '')}.dctl):",
            f"  Hue {center:.1f}°  Falloff {falloff:.1f}°  "
            f"{self.param_names[2]} {amount:+.3f}",
            f"  Zone {zone:+.3f}  Pivot {pivot:.3f}  "
            f"Luma Falloff {lfall:.3f}  Chroma {chroma:+.3f}",
        ])


class SectorSkewStage(_SectorStage):
    """Shift the hues of one picked sector (Chromogen Sector Skew).
    Amount in degrees of hue shift at full window weight."""

    name = "Sector Skew"
    param_names = ["Hue", "Falloff", "Skew",
                   "Zone", "Pivot", "Luma Falloff", "Chroma"]
    _AMOUNT_LO, _AMOUNT_HI = -60.0, 60.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        w = self._weight(hue, sat, val, params)
        hue2 = hue + w * (params[2] / 360.0)
        return reuleaux_to_rgb(np.stack([hue2, sat, val], axis=-1))



class SectorBrightnessStage(_SectorStage):
    """Brighten/darken one picked sector (Chromogen Sector Brightness).
    Val effect scales with sat (reuleaux convention) so the neutral
    axis is untouched."""

    name = "Sector Brightness"
    param_names = ["Hue", "Falloff", "Brightness",
                   "Zone", "Pivot", "Luma Falloff", "Chroma"]
    _AMOUNT_LO, _AMOUNT_HI = -3.0, 3.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        w = self._weight(hue, sat, val, params)
        val2 = val * np.maximum(1.0 + sat * (w * params[2]), 1e-6)
        return reuleaux_to_rgb(np.stack([hue, sat, val2], axis=-1))



class SectorSaturationStage(_SectorStage):
    """Saturate/desaturate one picked sector (Chromogen Sector
    Saturation). Amount is the DCTL-style sat factor (1 = neutral)."""

    name = "Sector Saturation"
    param_names = ["Hue", "Falloff", "Saturation",
                   "Zone", "Pivot", "Luma Falloff", "Chroma"]
    _AMOUNT_ID = 1.0
    _AMOUNT_LO, _AMOUNT_HI = 0.05, 2.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        w = self._weight(hue, sat, val, params)
        sat2 = _spow(sat, 1.0 / (1.0 + w * (params[2] - 1.0)))
        return reuleaux_to_rgb(np.stack([hue, sat2, val], axis=-1))



class SectorSquashStage(_SectorStage):
    """Compress (or, negative, SPREAD) the hues of one sector toward /
    away from the picked hue (Chromogen Sector Squash):

        h' = T + delta * (1 - s_eff * w(delta))

    with w the cos^2 window over the falloff. For this window shape,
    s_eff in [-1, 1] is inherently foldover-proof: the hue transfer's
    slope stays >= 0 everywhere (equality only at full squash at the
    center), so hue crossings/banding are mathematically impossible.
    The modulation block gates s (and its chroma gate keeps noisy
    near-neutral hues out — that's what makes squash safe here)."""

    name = "Sector Squash"
    param_names = ["Hue", "Falloff", "Squash",
                   "Zone", "Pivot", "Luma Falloff", "Chroma"]
    _AMOUNT_LO, _AMOUNT_HI = -1.0, 1.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        target = (params[0] / 360.0) % 1.0
        width = max(params[1] / 360.0, 1e-6)

        delta = ((hue - target + 0.5) % 1.0) - 0.5
        t = np.clip(np.abs(delta) / width, 0.0, 1.0)
        w = np.cos(0.5 * np.pi * t) ** 2

        m = modulation(val, sat, params[3], params[4], params[5], params[6])
        s_eff = params[2] * m
        hue2 = target + delta * (1.0 - s_eff * w)
        return reuleaux_to_rgb(np.stack([hue2, sat, val], axis=-1))



CHROMOGEN_STAGES = [
    ColourSaturationStage,
    ContrastBoostStage,
    HighlightBleachStage,
    NeutralTintStage,
    ColourCrosstalkStage,
    SectorSkewStage,
    SectorBrightnessStage,
    SectorSaturationStage,
    SectorSquashStage,
]
