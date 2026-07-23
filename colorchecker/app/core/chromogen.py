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
Two tools deviate from the standard block, copied from Baselight's own
panels: Neutral Tint's Chroma is 0..2 with 1 = everything (2 = only
neutrals, 0 = only saturated), and Brilliance Reduction's Chroma/
Pivot/Falloff all live in the saturation domain.

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


# ----------------------------------------------------- auto-naming
# Human labels for fitted stages ("skew dark greens", "cool lows") so
# a solve reads top-level without opening the sliders (Marc: hue
# numbers in Resolve are a blind thing).

_HUE_WORDS = ["red", "orange", "yellow", "lime", "green", "teal",
              "cyan", "azure", "blue", "purple", "magenta", "pink"]


def hue_word(deg: float) -> str:
    return _HUE_WORDS[int(round((deg % 360.0) / 30.0)) % 12]


def zone_word(zone: float) -> str:
    if zone > 0.3:
        return "bright "
    if zone < -0.3:
        return "dark "
    return ""


def _chroma_note(chroma: float) -> str:
    if chroma > 0.3:
        return " (saturated only)"
    if chroma < -0.3:
        return " (muted only)"
    return ""


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
        # R/G and Y/B are 0..2 with the 1.0 identity dead-centre, as on
        # the Chromogen panel (Baselight's Extended Ranges would go
        # further; 2x sat is already a lot)
        lo = [0.0, 0.0, -1.0, -6.0, -1.0]
        hi = [2.0, 2.0, 1.0, 8.0, 1.0]
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

    def label(self, params):
        s_rg, s_yb, zone, pivot, chroma = params
        mean = (s_rg + s_yb) / 2.0
        if abs(mean - 1.0) < 0.05 and abs(s_rg - s_yb) < 0.05:
            return "saturation (idle)"
        verb = "boost" if mean > 1.0 else "desat"
        axis = ""
        if abs(s_rg - s_yb) >= 0.15:
            axis = " Y/B" if s_yb > s_rg else " G/M"
        return f"{verb} {zone_word(zone)}colors{axis}{_chroma_note(chroma)}"

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

    # The inherent brightness weighting yields to the Zone mask as
    # |Zone| rises: at full zone the luma selection comes ONLY from the
    # mask. Without this, zoning to the shadows multiplied the (tiny)
    # shadow val into the effect exactly where the mask pointed and the
    # tool visibly did nothing at full throw (Marc, 2026-07-21).

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
        inherent = (1.0 - abs(zone)) * val + abs(zone)
        c1, c2 = to_chroma_vec(hue, sat)
        for i in range(4):
            one_hot = np.zeros(4)
            one_hot[i] = 1.0
            w = rygb_interp(hue, one_hot) * sat * inherent * m * amounts[i]
            d1, d2 = _axis_dir(self._AXES[i])
            c1 = c1 + w * d1
            c2 = c2 + w * d2
        hue2, sat2 = from_chroma_vec(c1, c2)
        return reuleaux_to_rgb(np.stack([hue2, sat2, val], axis=-1))

    def label(self, params):
        amounts = np.asarray(params[:4], dtype=np.float64)
        if np.abs(amounts).max() < 0.05:
            return "crosstalk (idle)"
        if np.abs(amounts - amounts.mean()).max() < 0.05:
            return ("global twist +" if amounts.mean() > 0
                    else "global twist -")
        i = int(np.abs(amounts).argmax())
        src = ["reds", "yellows", "greens", "blues"][i]
        toward = (["yellow", "blue"] if i in (0, 2)
                  else ["green", "magenta"])
        direction = toward[0] if amounts[i] > 0 else toward[1]
        return f"tilt {src} toward {direction}"

    def describe(self, params):
        r, y, g, b, zone, pivot, chroma = params
        return "\n".join([
            "Colour Crosstalk (paste into dctl/ColourCrosstalk.dctl):",
            f"  R -> Y/B {r:+.3f}  Y -> R/G {y:+.3f}  "
            f"G -> Y/B {g:+.3f}  B -> R/G {b:+.3f}",
            f"  Zone {zone:+.3f}  Pivot {pivot:+.3f}  Chroma {chroma:+.3f}",
        ])


class ContrastCurveStage(Stage):
    """Parametric film contrast curve (replaces the old soft-S "Contrast
    Boost"): a real toe + shoulder S with independent shadow/highlight
    shaping, a movable mid-point, and pre-curve exposure/flare. Modelled
    on the [1D] CONTRAST panel Marc grades with (Diachromie) — a
    behavioral reimplementation from his screenshots, not a source port.
    Built to sit UNDER the DRT (which carries its own curve), so every
    control is gentle and the defaults are an EXACT identity.

    All tonal work is in LogC3 code values, measured in STOPS from
    mid-grey. The pivot is fixed at mid-grey; sliders (Contour "Curve"
    model — independent toe & shoulder Length + Strength):
      Contrast          mid slope at the pivot (1 = identity, a straight
                        line; >1 steepens the mid, the ends roll toward the
                        white/black points and never hard-clip)
      Black Point       shadow asymptote LEVEL, in stops below mid-grey —
                        the curve approaches it, never reaches it. Identity
                        is far out (-10, no toe in range); bring it in for
                        deeper (or, above -~5, lifted) blacks.
      White Point       highlight asymptote level, in stops above mid-grey
      Toe Length        how long the mid stays linear before the toe rolls:
                        1 = knee at the black point (no toe in range),
                        lower starts the toe earlier
      Toe Strength      toe knee sharpness: 0 = a gentle round, 1 = stays
                        straight longer then a tighter corner
      Shoulder Length   the same, for the highlight shoulder
      Shoulder Strength shoulder knee sharpness
      Preserve Color    0 = per-RGB (contrast raises saturation, the film
                        look), 1 = luma only (chromaticity preserved)
      Mid Push          a midtone bump; + lifts mids, - drops them
      Mid Compensate    0 = the bump lifts the pivot (a midtone exposure);
                        1 = the pivot is held and the bump becomes a local
                        S — the touch of mid contrast film's 'straight
                        line' really has (K64)
      Blend             master mix with the untouched input
      Flare             milky shadow lift, tapering to identity by the highs
      Exposure          overall shift in stops, applied AFTER the curve
                        (mid-grey referenced, achromatic — see _expose)

    Black/White Point are the asymptote LEVELS in stops from mid-grey, kept
    to a sensible visible range: -6 stops is ~code 0 (crushed), Marc's
    baseline black is ~-4.2 (~code 0.08), -1.5 is milky; drive the toe
    below the LogC3 floor and the shadows just clamp, so the range stops
    there. Length is the roll's EXTENT: 0 = knee at the point (no roll in
    range = identity), 1 = knee at the pivot (the whole end rolls).
    """

    name = "Contrast Curve"
    # NOTE the order: the 12 FLOAT sliders map 1:1 onto the DCTL's
    # sliderFloatParam0..11 and the .drx patch; "Mid Compensate" is LAST
    # because in the DCTL it is a CHECK_BOX (not a sliderFloatParam), so
    # keeping it out of the float run keeps every other slider aligned.
    param_names = [
        "Contrast", "Black Point", "White Point",
        "Toe Length", "Toe Strength", "Shoulder Length", "Shoulder Strength",
        "Preserve Color", "Mid Push", "Blend", "Flare", "Exposure",
        "Mid Compensate",
    ]

    # curve shape constants (stops unless noted)
    _EPS = 1e-6         # guards the headroom divide when Length -> 0
    _STR_GAIN = 8.0     # Strength 0->1 maps the knee exponent n 1..9 (sharp)
    _MID_W = 2.0        # mid bump/S half-width
    _MID_SCALE = 1.0    # Mid Push max lift (stops) at Push = 1
    _FLARE_SCALE = 0.045  # code-value shadow lift per Flare unit
    _FLARE_WIDTH = 3.0    # how far up (stops) the flare lift fades out

    def identity(self):
        # Length 0 parks both knees AT the points (out of the working
        # range), so identity is an EXACT straight line; the solver / a
        # grade raises a Length to roll that end in.
        return np.array([1.0, -6.0, 12.0, 0.0, 0.5, 0.0, 0.5,
                         0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def init(self):
        # Start the SOLVE with the toe & shoulder mid-engaged (Length 0.5)
        # so the fit has a live gradient on the Point/Length/Strength
        # controls (the reg still anchors at identity(), so an unneeded
        # roll relaxes out). Black Point ~-4.5 seeds a sensible film black
        # (toe lands ~code 0.06, near the LogC3 floor / Marc's baseline).
        return np.array([1.0, -4.5, 8.0, 0.5, 0.5, 0.5, 0.5,
                         0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def bounds(self):
        # Black/White Point clamped to the visible range (see class doc);
        # driving them further just clamps shadows/highlights and the
        # sliders would appear dead.
        lo = [0.2, -6.0,  2.5, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -3.0, 0.0]
        hi = [3.0, -1.5, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0,  2.0, 1.0, 2.0,  3.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    # ---- the scalar tone pipeline, run on val or on each RGB channel --

    @staticmethod
    def _gsc(u, n):
        """Generalized soft-clip: slope 1 at 0, asymptotes to +-1, with a
        knee sharpness `n` (n->1 round, large n -> linear then a corner)."""
        return u / np.power(1.0 + np.power(np.abs(u), n), 1.0 / n)

    def _curve(self, s, contrast, bp, wp, toe_len, toe_str, sh_len, sh_str):
        """A BOUNDED film S with INDEPENDENT toe & shoulder (Contour model).
        Mid slope at the pivot is `contrast`; each end then rolls smoothly
        toward its asymptote — the White/Black Point level in stops — with
        the roll starting at a knee whose POSITION is set by Length and
        whose SHARPNESS by Strength. C1-continuous, never hard-clips. With
        the knees past the working range (Length 1, ±10-stop points) it is
        an EXACT straight line, so contrast 1 is exact identity.
        """
        y = contrast * s
        # shoulder (highlights): Length 0 -> knee at the point (no roll in
        # range), Length 1 -> knee at the pivot (the whole end rolls)
        yk_hi = (1.0 - sh_len) * wp
        h_hi = wp - yk_hi + self._EPS                 # headroom (> 0)
        n_hi = 1.0 + self._STR_GAIN * sh_str
        e_hi = np.maximum(y - yk_hi, 0.0)
        hi = yk_hi + h_hi * self._gsc(e_hi / h_hi, n_hi)
        # toe (shadows): bp < 0, so the knee and headroom are negative
        yk_lo = (1.0 - toe_len) * bp
        h_lo = bp - yk_lo - self._EPS                 # headroom (< 0)
        n_lo = 1.0 + self._STR_GAIN * toe_str
        e_lo = np.minimum(y - yk_lo, 0.0)
        lo = yk_lo + h_lo * self._gsc(e_lo / h_lo, n_lo)
        return np.where(y > yk_hi, hi, np.where(y < yk_lo, lo, y))

    def _midterm(self, s, mid_push, comp):
        u = s / self._MID_W
        g = np.exp(-0.5 * u * u)
        hump = g                             # symmetric: lifts the pivot
        scurve = u * np.exp(0.5) * g         # antisymmetric: holds the pivot
        shape = (1.0 - comp) * hump + comp * scurve
        return mid_push * self._MID_SCALE * shape

    @staticmethod
    def _expose(x, exposure):
        """A purely tonal (greyscale) exposure move, mid-grey referenced:
        slide every pixel along the Reuleaux value axis by `exposure` stops
        (so mid-grey moves by exactly that many stops) and preserve its
        hue/chroma, so exposure NEVER recolours. Applied AFTER the tone
        curve — the curve is free to reshape tonality; exposure just
        repositions the result on the luma axis."""
        if exposure == 0.0:
            return x
        r = rgb_to_reuleaux(x).copy()
        r[..., 2] = r[..., 2] + exposure * STOP
        return reuleaux_to_rgb(r)

    def _tone(self, v, params):
        (contrast, bp, wp, toe_len, toe_str, sh_len, sh_str,
         _preserve, mid_push, _blend, flare, _exposure, comp) = params
        # exposure is handled achromatically AFTER the curve in apply()
        # (see _expose), NOT here — the per-channel curve would tint it.
        shadow_w = 1.0 - ramp_window(v, MID_GREY, self._FLARE_WIDTH * STOP)
        v = v + flare * self._FLARE_SCALE * shadow_w
        s = (v - MID_GREY) / STOP
        y = (self._curve(s, contrast, bp, wp, toe_len, toe_str, sh_len, sh_str)
             + self._midterm(s, mid_push, comp))
        return MID_GREY + y * STOP

    def apply(self, x, params):
        preserve, blend = params[7], params[9]     # Preserve Color = luma blend
        rgb_out = self._tone(x, params)
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        luma_out = reuleaux_to_rgb(np.stack(
            [hue, sat, self._tone(val, params)], axis=-1))
        curved = (1.0 - preserve) * rgb_out + preserve * luma_out
        curved = self._expose(curved, params[11])  # exposure AFTER the curve
        return (1.0 - blend) * x + blend * curved

    def label(self, params):
        (contrast, bp, wp, toe_len, toe_str, sh_len, sh_str,
         preserve, mid_push, blend, flare, exposure, comp) = params
        if blend < 0.02:
            return "contrast (idle)"
        bits = []
        if contrast > 1.05:
            bits.append("add contrast")
        elif contrast < 0.95:
            bits.append("flatten contrast")
        if abs(exposure) > 0.05:
            bits.append("brighten" if exposure > 0 else "darken")
        if flare > 0.05:
            bits.append("lift shadows")
        if toe_len > 0.15 and bp > -3.0 and not bits:
            bits.append("lift blacks")
        elif toe_len > 0.15 and bp < -5.0 and not bits:
            bits.append("crush blacks")
        if not bits:
            return "contrast (idle)"
        note = " (rich)" if preserve < 0.25 else (
            " (clean)" if preserve > 0.75 else "")
        return bits[0] + note

    def describe(self, params):
        (contrast, bp, wp, toe_len, toe_str, sh_len, sh_str,
         preserve, mid_push, blend, flare, exposure, comp) = params
        return "\n".join([
            "Contrast Curve (paste into dctl/ContrastCurve.dctl):",
            f"  Contrast {contrast:.3f}  Black Point {bp:+.3f}  "
            f"White Point {wp:+.3f}",
            f"  Toe Length {toe_len:.3f}  Toe Strength {toe_str:.3f}  "
            f"Shoulder Length {sh_len:.3f}  Shoulder Strength {sh_str:.3f}",
            f"  Preserve Color {preserve:.3f}  Mid Push {mid_push:+.3f}  "
            f"Blend {blend:.3f}  Flare {flare:.3f}  Exposure {exposure:+.3f}",
            f"  Mid Compensate (checkbox) {comp:.0f}",
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
        # Chromogen panel defaults: Pivot -2.00 (stops below mid-grey,
        # confirmed by the knob position on the panel's grey bar) and
        # Falloff 0.500 — a soft-kneed threshold: everything above ~2
        # stops under mid-grey bleaches once an amount is raised
        return np.array([0.0, 0.0, 0.0, 0.0, -2.0, 0.5, 0.0])

    def bounds(self):
        lo = [0.0, 0.0, 0.0, 0.0, -6.0, 0.1, -1.0]
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

    def label(self, params):
        amounts = np.asarray(params[:4], dtype=np.float64)
        if amounts.max() < 0.03:
            return "bleach (idle)"
        spared = [w for a, w in zip(amounts, ["reds", "yellows",
                                              "greens", "blues"])
                  if a < 0.4 * amounts.max()]
        note = f" (spare {', '.join(spared)})" if spared else ""
        return f"bleach highlights{note}"

    def describe(self, params):
        r, y, g, b, pivot, falloff, chroma = params
        return "\n".join([
            "Highlight Bleach (paste into dctl/HighlightBleach.dctl):",
            f"  R {r:.3f}  Y {y:.3f}  G {g:.3f}  B {b:.3f}",
            f"  Pivot {pivot:+.3f}  Falloff {falloff:.3f}  Chroma {chroma:+.3f}",
        ])


class NeutralTintStage(Stage):
    """Neutral Tint v4 — bell falloff (Marc's design, 2026-07-23): ONE
    gaussian bell of weight laid over the log range replaces the v3
    one-sided ramp. The ramp saturated at full strength below (or
    above) its knee — deep blacks pinned at maximum tint, the "funky
    in the bottom" behaviour; the bell rolls off on BOTH sides, so the
    extremes fall back toward untinted on their own. Still the
    SUM-PRESERVING RGB offset in LOG toward the picked hue (zero-sum
    direction, channel mean = log exposure untouched).

    Amount is SIGNED with 0 = identity, and does the one-slider
    gesture: |Amount| is the tint strength AND Amount slides the bell
    centre (right = toward highlights, left = shadows; SHIFT_SCALE
    stops at full throw). Pivot re-anchors the bell centre (stops from
    mid-grey), which keeps every (position, strength) pair reachable —
    strong tint AT mid-grey = Amount full throw + Pivot pulled the
    other way. Falloff is the bell's sigma in STOPS: width and
    smoothness in one knob (a gaussian's width IS its smoothness).
    Chroma is the Baselight sat mask, unchanged: 1 = everything
    (default), down to 0 = only saturated colors, up to 2 = only
    neutrals. (NOTE: this differs from the signed 0-centred Chroma of
    the other tools — copied from Baselight's panel deliberately.)"""

    name = "Neutral Tint"
    param_names = ["Hue", "Amount", "Pivot", "Falloff", "Chroma"]

    # slider +-1.0 maps to an RGB offset of TINT_SCALE (L2, code
    # values): full throw on deep shadows separates channels by up to
    # ~1.6 stops — strong but usable; mid-slider is a subtle tint
    TINT_SCALE = 0.15
    # stops the bell centre travels at full Amount throw — eyeball
    # guess (full right = centred 3 stops up, well into highlights)
    SHIFT_SCALE = 3.0

    def identity(self):
        # bell anchored on mid-grey, sigma 2 stops — a broad tonal
        # band once Amount engages; Falloff narrows/widens from there
        return np.array([0.0, 0.0, 0.0, 2.0, 1.0])

    def bounds(self):
        # Falloff floor 0.5: the bell is smooth by construction at any
        # width (v3's 1.0-stop floor guarded the ramp turning into a
        # step), below ~0.5 sigma the band just gets surgical. Pivot
        # floor -4 kept from v3 (below ~-4.2 stops is sub-black code);
        # with Amount full left the centre still reaches -7.
        lo = [0.0, -1.0, -4.0, 0.5, 0.0]
        hi = [360.0, 1.0, 8.0, 8.0, 2.0]
        return np.asarray(lo), np.asarray(hi)

    @staticmethod
    def _tint_direction(hue_deg):
        """Unit (L2) zero-sum RGB direction whose reuleaux hue is
        hue_deg — the inverse of the rgb->rot rotation restricted to
        the chroma plane; the un-normalized vector has norm sqrt(3)
        for every hue."""
        ang = (hue_deg / 360.0) * _TWO_PI
        s2, s6 = np.sqrt(2.0), np.sqrt(6.0)
        d = np.array([
            s2 * np.cos(ang),
            (s6 * np.sin(ang) - s2 * np.cos(ang)) / 2.0,
            (-s6 * np.sin(ang) - s2 * np.cos(ang)) / 2.0,
        ])
        return d / np.sqrt(3.0)

    def apply(self, x, params):
        hue_deg, amount, pivot, falloff, chroma = params
        x = np.asarray(x, dtype=np.float64)
        reuleaux = rgb_to_reuleaux(x)
        sat, val = reuleaux[..., 1], reuleaux[..., 2]

        centre = pivot + self.SHIFT_SCALE * amount   # stops from mid-grey
        z = ((val - MID_GREY) / STOP - centre) / falloff
        bell = np.exp(-0.5 * z * z)
        m = bell * modulation(val, sat, 0.0, 0.0, 1.0 - chroma)

        strength = abs(amount) * self.TINT_SCALE
        return x + (strength * m)[..., None] * self._tint_direction(hue_deg)

    def label(self, params):
        hue_deg, amount, pivot, falloff, chroma = params
        if abs(amount) < 0.03:
            return "tint (idle)"
        h = hue_deg % 360.0
        if h < 90.0 or h >= 330.0:
            tone = "warm"
        elif 150.0 <= h < 270.0:
            tone = "cool"
        else:
            tone = hue_word(h)
        centre = pivot + self.SHIFT_SCALE * amount
        where = "highs" if centre >= 0.5 else "lows" if centre <= -0.5 else "mids"
        return f"{tone} {where}"

    def describe(self, params):
        hue_deg, amount, pivot, falloff, chroma = params
        centre = pivot + self.SHIFT_SCALE * amount
        return "\n".join([
            "Neutral Tint (paste into dctl/NeutralTint.dctl):",
            f"  Hue {hue_deg:.1f}°  Amount {amount:+.3f}",
            f"  Pivot {pivot:+.3f}  Falloff {falloff:.3f}  "
            f"Chroma {chroma:.3f} (1 = all, 0 = saturated, 2 = neutrals)",
            f"  -> bell centre {centre:+.2f} stops from mid-grey, "
            f"sigma {falloff:.2f} stops",
        ])


class BrillianceReductionStage(Stage):
    """Baselight Brilliance Reduction (the last Chromogen-family tool):
    darken colors ACCORDING TO their saturation — a plain luminance
    scale (chromaticity untouched), weighted by a sat-domain ramp.
    Chroma, Pivot and Falloff all live in the SATURATION domain
    (reuleaux sat units, not stops): Pivot is where the ramp starts
    biting, Falloff its width, Chroma the overall mask strength.

    Amount 0.0 (LEFT end) is the identity — raise it to reduce. The
    reduction is an EXPOSURE scale in stops, 2^(-REDUCTION_STOPS *
    amount * mask): even the most pathological settings (amount 1,
    chroma 1, pivot 0, falloff 0 = mask everywhere) bottom out at
    -REDUCTION_STOPS, never black — a linear scale could hit exactly 0
    and crushed the image (Marc, 2026-07-21). (Amount correction same
    day: the first screenshot showed a non-default grade with Amount
    at 1.0; Baselight's true default is 0 and identity-at-1 read as a
    dead panel.) Defaults Chroma 0.6 / Pivot 0.35 / Falloff 0.5 shape
    the mask but do nothing while Amount stays at 0."""

    name = "Brilliance Reduction"
    param_names = ["Amount", "Chroma", "Pivot", "Falloff"]

    # full throw with a fully-open mask darkens by exactly this many
    # stops — the ceiling that keeps the tool gentle by construction
    REDUCTION_STOPS = 2.0

    def identity(self):
        return np.array([0.0, 0.6, 0.35, 0.5])

    def bounds(self):
        lo = [0.0, 0.0, 0.0, 0.01]
        hi = [1.0, 1.0, 1.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        amount, chroma, pivot, falloff = params
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        w = chroma * ramp_window(sat, pivot, falloff)
        val2 = val * 2.0 ** (-self.REDUCTION_STOPS * amount * w)
        return reuleaux_to_rgb(np.stack([hue, sat, val2], axis=-1))

    def label(self, params):
        amount = params[0]
        if amount < 0.05:
            return "brilliance (idle)"
        return "reduce brilliance"

    def describe(self, params):
        amount, chroma, pivot, falloff = params
        return "\n".join([
            "Brilliance Reduction (paste into dctl/BrillianceReduction.dctl):",
            f"  Amount {amount:.3f}",
            f"  Chroma {chroma:.3f}  Pivot {pivot:.3f}  Falloff {falloff:.3f}"
            "  (all in the sat domain)",
        ])


# ------------------------------------------------------ sector family

class _SectorStage(Stage):
    """Shared machinery for the single-picked-hue sector tools: a
    wrapped cos^2 hue window (Hue + Falloff, degrees) times the
    standard Zone/Pivot/Chroma modulation gates one adjustment.
    Param vector: [hue, amount, falloff, zone, pivot, chroma]."""

    # subclasses override index 1 with their tool's slider name
    param_names = ["Hue", "Amount", "Falloff", "Zone", "Pivot", "Chroma"]
    local_tool = True

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

    def label(self, params):
        hue, skew, falloff, zone = params[0], params[1], params[2], params[3]
        if abs(skew) < 2.0:
            return "skew (idle)"
        toward = hue_word(hue + np.sign(skew) * 45.0)
        return (f"skew {zone_word(zone)}{hue_word(hue)}s "
                f"toward {toward}")

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

    def label(self, params):
        if abs(params[1]) < 0.05:
            return "brightness (idle)"
        verb = "brighten" if params[1] > 0 else "darken"
        return f"{verb} {zone_word(params[3])}{hue_word(params[0])}s"

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
    Saturation). LINEAR chroma scale (1 = neutral, 0 = full desat):
    constant relative gain, so low-sat noise is
    amplified no more than the colors are — the earlier power-law
    version had unbounded relative gain near the neutral axis and
    visibly amplified sensor noise (Marc, confirmed on footage)."""

    name = "Sector Saturation"
    param_names = ["Hue", "Saturation", "Falloff", "Zone", "Pivot", "Chroma"]

    def label(self, params):
        if abs(params[1] - 1.0) < 0.05:
            return "sector sat (idle)"
        verb = "boost" if params[1] > 1.0 else "desat"
        return f"{verb} {zone_word(params[3])}{hue_word(params[0])}s"

    _AMOUNT_ID = 1.0
    _AMOUNT_LO, _AMOUNT_HI = 0.0, 2.0

    def apply(self, x, params):
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]
        w = self._weight(hue, sat, val, params)
        sat2 = sat * (1.0 + w * (params[1] - 1.0))
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

    def label(self, params):
        if abs(params[1]) < 0.05:
            return "squash (idle)"
        verb = "squash" if params[1] > 0 else "spread"
        return f"{verb} {zone_word(params[3])}{hue_word(params[0])}s"

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


class SplitToneStage(Stage):
    """Split Tone v5 — a per-channel SINGLE sextic-Bezier tone shaper.

    v1/v2 (ported from Bezier_Split_Tone_V2.dctl) used TWO Beziers per
    channel joined at a pivot crossover; the joint was only C0 — a
    derivative kink that showed as BANDING on footage (Marc,
    2026-07-22: "each rgb channel needs to stay super smooth!!"). v3
    replaced that with ONE cubic Bezier per channel; v4 (2026-07-23)
    added the Mid handle for independent mid-grey shaping. v5 (same
    day, Marc reviewing the genesis fit: "there is one crossover in
    the shoulder ... that we simply cannot replicate like this,
    should be the same in the toe") adds DARK and SPECULAR handles — one
    degree-6 Bezier with y-handles at exact sixth x-positions, so it
    reduces to a smooth sextic polynomial y(v) — C-infinity, no
    pivot, no joins. Two handles between Mid and each endpoint let a
    channel-difference change SIGN in the toe/shoulder (genesis: the
    R-G crossover at +6.2 stops was unreachable with one). Outside
    [0,1] the curve continues LINEARLY with the endpoint slope (C1),
    so super-whites/blacks never snap.

    Per channel, low to high — NAMED BY ON-SCREEN IMPACT (each
    handle's Bernstein basis peaks at code x; through plain openDRT
    that lands at these display levels — Marc 2026-07-23, "relate the
    X coordinate to the appearance on screen"):
      Black  (x=0    -> display   0%)   Dark     (x=1/6 -> 10%)
      Low Mid  (x=2/6 -> 29%)           High Mid (x=3/6 -> 59%)
      Highlight (x=4/6 -> 86%)          Specular (x=5/6 -> 97%)
      White  (x=1    -> 100%)
    The old Shadow/Mid names sat one band too low (they straddle mid
    grey). Defaults
    Black 0 / everything else 1 are an EXACT identity (the Bezier of
    a straight line IS the straight line). Moving one channel against
    another IS the split tone. Works on raw code values — no transfer
    function involved anymore.

    NOTE v4 values do NOT transfer 1:1 (handles moved from quarters
    to sixths); JSON presets by name still load — the new "Dark" and
    "Specular" default to identity.
    """

    name = "Split Tone"
    param_names = [
        "Black R", "Black G", "Black B",
        "Dark R", "Dark G", "Dark B",
        "Low Mid R", "Low Mid G", "Low Mid B",
        "High Mid R", "High Mid G", "High Mid B",
        "Highlight R", "Highlight G", "Highlight B",
        "Specular R", "Specular G", "Specular B",
        "White R", "White G", "White B",
    ]

    _BIN6 = (1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0)

    def identity(self):
        return np.array([0.0] * 3 + [1.0] * 18)

    def bounds(self):
        # widened +-0.5 (Marc 2026-07-23: "a lot of the split tone
        # sliders are maxed out") — fits kept pinning 0 / 2
        lo = [-1.5] * 3 + [-0.5] * 18
        hi = [1.5] * 3 + [2.5] * 18
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        for ci in range(3):
            y = (params[0 + ci] / 6.0,               # Black  (0)
                 params[3 + ci] / 6.0,               # Dark   (1/6)
                 params[6 + ci] / 3.0,               # Low Mid (2/6)
                 params[9 + ci] * 0.5,               # High Mid (3/6)
                 1.0 - (2.0 - params[12 + ci]) / 3.0,  # Highlight (4/6)
                 1.0 - (2.0 - params[15 + ci]) / 6.0,  # Specular (5/6)
                 params[18 + ci])                    # White  (1)
            v = x[..., ci]
            vc = np.clip(v, 0.0, 1.0)
            iv = 1.0 - vc
            bez = sum(self._BIN6[k] * y[k] * iv ** (6 - k) * vc ** k
                      for k in range(7))
            lo_ext = y[0] + 6.0 * (y[1] - y[0]) * v      # C1 tails
            hi_ext = y[6] + 6.0 * (y[6] - y[5]) * (v - 1.0)
            out[..., ci] = np.where(v < 0.0, lo_ext,
                                    np.where(v > 1.0, hi_ext, bez))
        return out

    def label(self, params):
        p = np.asarray(params, dtype=float)
        if np.max(np.abs(p[0:3])) < 0.03 \
                and np.max(np.abs(p[3:21] - 1.0)) < 0.03:
            return "split (idle)"
        # crude warm/cool read from the R-vs-B shadow balance
        lo = (p[0] - p[2]) + (p[3] - p[5])
        return "warm lows" if lo > 0 else "cool lows"

    def describe(self, params):
        p = np.round(np.asarray(params, dtype=float), 3)
        return "\n".join([
            "Split Tone v5 (sextic; DCTL not updated yet):",
            f"  Black   R {p[0]}  G {p[1]}  B {p[2]}",
            f"  Dark    R {p[3]}  G {p[4]}  B {p[5]}",
            f"  Low Mid R {p[6]}  G {p[7]}  B {p[8]}",
            f"  High Mid R {p[9]}  G {p[10]}  B {p[11]}",
            f"  Highlight R {p[12]}  G {p[13]}  B {p[14]}",
            f"  Specular R {p[15]}  G {p[16]}  B {p[17]}",
            f"  White   R {p[18]}  G {p[19]}  B {p[20]}",
        ])


CHROMOGEN_STAGES = [
    ColourSaturationStage,
    ColourCrosstalkStage,
    ContrastCurveStage,
    HighlightBleachStage,
    SplitToneStage,
    NeutralTintStage,
    BrillianceReductionStage,
    SectorSkewStage,
    SectorBrightnessStage,
    SectorSaturationStage,
    SectorSquashStage,
]

# NeutralTintStage stays a known stage (STAGE_POOL: presets, manual bake,
# .drx node mapping) but is EXCLUDED from the ML search's audition pool —
# Split Tone replaces it for fitting (Marc, 2026-07-22). See
# chain_search.default_pool().
