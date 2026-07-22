"""1:1 Python port of ME_Filmic Contrast V1.3 (Moaz Elgabry).

Source: Marc's installed FilmicContrast.dctl (committed at
dctl/FilmicContrast.dctl), transcribed function-for-function
(vectorized, float64) like the reuleaux and openDRT ports. The tool is
Marc's pick for the chain's tone node ("for the contrast, lets just use
this for now, this really works").

Scope of the port (the PIXEL path only):
  * transfer function fixed to ARRI LogC3 (combo 0) — Marc's pipeline;
  * Preserve Mid-gray toggle ON (the DCTL default);
  * Tone-mapped exposure toggle OFF (default): exposure is a linear
    gain applied at the END — achromatic in linear light and mid-grey
    referenced (+N stops moves mid-grey exactly N stops), which is the
    behaviour Marc required of the tone node's Exposure;
  * screen furniture (ramp / curve / checker overlays, bypass) is NOT
    ported — it never touches the measurement path.

Deliberate deviations from the DCTL (each is the mathematical limit,
made exact so the Stage identity contract holds and the solver can't
produce NaNs):
  * White Point raw 1.02 (sanitized == 1.0) and Black Point raw 0.0
    (sanitized == 1.0) switch their sections to exact passthrough —
    the DCTL's formulas approach identity there but divide by zero AT
    it. 1.02 sits just past the stock UI max 1.018 (== 0.999
    sanitized, visually identical); our themed DCTL copy widens the
    slider to match.
  * White Point extended DOWN (slider min -0.15, stock 0.0): the two
    sanitize floors (0.5 pre, 0.7 in preserve-midgray) relaxed to the
    pivot*1.1 safety limit — stock dead-zoned the slider below ~0.42;
    the white ceiling can now fade down to just above mid-grey.
  * Black Point slider range extended to 1.5 (stock caps at 0.5 —
    Marc, 2026-07-22: "extend the range to 1, maybe 1.5"). The stock
    sanitize floors the internal point at 0.69, which silently
    dead-zones the slider past ~0.775, so the floor is lowered to 0.4
    (raw 1.5 lands exactly on it). Mirrored in our DCTL copy.
  * Exposure == 0 / Pop Mids == 0 / Flare == 0 skip their
    LogC3<->linear round-trips (a *1.0 gain / +0.0 offset changes
    nothing mathematically; skipping only avoids float round-trip
    noise so identity is exact).
  * shoulder/toe roll ratios are floored at 1e-3 before the fractional
    power (float32-safe at strength 10; same guard added to our DCTL):
    where the stock tool NaNs (roll pivot beyond the scaled point) the
    limit flattens that end at the pivot instead of breaking the image.
  * Shoulder Falloff extended to 9.7 (reaches the softness floor) and
    the Shoulder slider step fixed 0.1 -> 0.001; Toe Falloff remapped
    in preserve-midgray from strength 1.48..2.95 to 0.25..3.35 (much
    wider toe shapes, the default 2 -> 2.65 unchanged). Marc, 22 eve.
  * Pop Mids is mid-grey COMPENSATED (not in the stock tool): a global
    counter-gain re-anchors mid-grey exactly where the tone block put
    it, decoupling Pop Mids from Exposure (Marc: "compensate exposure
    to keep mid grey intact").
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- LogC3

_A = 5.555556
_B = 0.052272
_C = 0.247190
_D = 0.385537
_E = 5.367655
_F = 0.092809
_CUT = 0.010591


def lin_to_logc3(x):
    """Mirror of LinToLogC3 (per component)."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):   # x <= cut -> log branch discarded
        return np.where(x > _CUT,
                        _C * np.log10(_A * x + _B) + _D, _E * x + _F)


def logc3_to_lin(x):
    """Mirror of LogC3ToLin."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > _E * _CUT + _F,
                    (10.0 ** ((x - _D) / _C) - _B) / _A,
                    (x - _F) / _E)


#: LinToLogC3(0.18) — what set_pivot actually uses (the table constant
#: 0.39101 in the DCTL is computed then overwritten by this).
MID_GREY = float(lin_to_logc3(0.18))


def set_pivot(pivot: float) -> float:
    """Mirror of set_pivot for LogC3: relative slider -> absolute code."""
    return float(np.clip(pivot + MID_GREY, 0.02, 1.0))


# ------------------------------------------------------------- helpers

def powerf(base, exp):
    """Sign-preserving power, as in the DCTL."""
    base = np.asarray(base, dtype=np.float64)
    return np.sign(base) * np.abs(base) ** exp


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 3.0 * x ** 2 - 2.0 * x ** 3


def smootherstep5(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def gentle_ramp(x):
    """Mirror of gentleRamp: smootherstep then cube (soft onset)."""
    return smootherstep(np.clip(x, 0.0, 1.0)) ** 3


def soft_union(a, b):
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    return 1.0 - (1.0 - a) * (1.0 - b)


def gauss_weight(x, mu, sigma):
    s = max(abs(sigma), 1e-6)
    d = (x - mu) / s
    return float(np.exp(-0.5 * d * d))


def remap_asym_signed(u, neg_min, pos_max):
    """Mirror of remapAsymSigned (signed square easing)."""
    u = float(np.clip(u, -1.0, 1.0))
    s = u * abs(u)
    return s * -neg_min if s < 0.0 else s * pos_max


# ------------------------------------------------- contrast primitives

def apply_linear_contrast(x, contrast, pivot):
    return (x - pivot) * contrast + pivot


def apply_rolling_contrast(x, contrast, pivot):
    """Mirror of apply_rolling_contrast: the two power halves with a
    C2 smootherstep5 blend window around the pivot."""
    x = np.asarray(x, dtype=np.float64)
    p = float(np.clip(pivot, 1e-6, 1.0 - 1e-6))
    e = max(0.01 * min(p, 1.0 - p), 1e-4)
    x0, x1 = p - e, p + e

    f_left = powerf(x / p, contrast) * p
    pm = 1.0 - p
    f_right = 1.0 - powerf((1.0 - x) / pm, contrast) * pm

    t = smootherstep5((x - x0) / max(x1 - x0, 1e-6))
    blend = f_left * (1.0 - t) + f_right * t
    return np.where(x <= x0, f_left, np.where(x >= x1, f_right, blend))


def apply_power_sigmoid_contrast(x, contrast, pivot):
    """Mirror of apply_power_sigmoid_contrast (PowerP piecewise
    sigmoid; the s0 == 1 singular limit is identity)."""
    x = np.asarray(x, dtype=np.float64)
    t0 = float(np.clip(pivot, 1e-4, 1.0 - 1e-4))

    s0 = float(np.clip(contrast, 0.5, 2.5))
    if s0 > 1.0:
        t = (s0 - 1.0) / (2.5 - 1.0)
        s0 *= 1.0 + 0.12 * t

    if s0 < 1.0:
        x = np.clip(x, 0.0, 1.0)
    if abs(s0 - 1.0) < 1e-5:
        return x.copy()

    eps = 1e-6
    denom = s0 - 1.0
    denom = np.sign(denom) * max(abs(denom), eps)
    s1 = s0 * (1.0 - t0) / denom
    s2 = s0 * (t0 - 0.0) / denom
    s1 = np.sign(s1) * max(abs(s1), eps) if s1 != 0.0 else eps
    s2 = np.sign(s2) * max(abs(s2), eps) if s2 != 0.0 else eps

    d_hi = x - t0
    hi = s0 * d_hi / (d_hi * s0 / s1 + 1.0) + t0
    d_lo = t0 - x
    lo = -s0 * d_lo / (d_lo * s0 / s2 + 1.0) + t0
    return np.where(x >= t0, hi, lo)


# ------------------------------------------- white / black point rolls

def _end_roll(v, point, pivot, strength):
    """The shared shoulder/toe compressor for values above `pivot`
    (mirror of the per-channel body of apply_white_point). `point` is
    where the roll lands at input 1.0. Base floored at 1e-3 (matching
    the guard added to our DCTL copy — float32-safe at strength 10):
    where the stock DCTL NaNs (point <= pivot) the limit flattens the
    end at the pivot."""
    base = max((point - pivot) / (1.0 - pivot), 1e-3)
    scale = (1.0 - pivot) / (base ** -strength - 1.0) ** (1.0 / strength)
    d = (v - pivot) / scale
    with np.errstate(invalid="ignore"):   # v < pivot -> NaN, discarded
        rolled = pivot + scale * d / (1.0 + d ** strength) ** (1.0 / strength)
    return np.where(v > pivot, rolled, v)


def apply_white_point(x, white_point, pivot, shoulder_str):
    """Mirror of apply_white_point. Sanitized white_point >= 1.0 is the
    exact-identity limit (the DCTL divides by zero there)."""
    x = np.asarray(x, dtype=np.float64)
    if white_point >= 1.0:
        return x.copy()
    return _end_roll(x, white_point, pivot, shoulder_str)


def apply_black_point(x, black_point, pivot, toe_str):
    """Mirror of apply_black_point: the same roll on the flipped image."""
    x = np.asarray(x, dtype=np.float64)
    if black_point >= 1.0:
        return x.copy()
    return 1.0 - _end_roll(1.0 - x, black_point, pivot, toe_str)


# ------------------------------------------------ CHEN colour model

_SQRT3 = 1.7320508075688772


def rgb_to_chen(rgb):
    """Mirror of RGBtoCHEN (Kaur Hendrikson): spherical coordinates
    around the grey axis -> (h, c, l)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rtr = r * 0.81649658 + g * -0.40824829 + b * -0.40824829
    rtg = g * 0.70710678 + b * -0.70710678
    rtb = r * 0.57735027 + g * 0.57735027 + b * 0.57735027

    art = np.arctan2(rtg, rtr)
    sphr = np.sqrt(rtr * rtr + rtg * rtg + rtb * rtb)
    spht = np.where(art < 0.0, art + 2.0 * 3.141592653589, art)
    sphp = np.arctan2(np.sqrt(rtr * rtr + rtg * rtg), rtb)

    return np.stack([spht * 0.15915494309189535,
                     sphp * 1.0467733744265997,
                     sphr * 0.5773502691896258], axis=-1)


def chen_to_rgb(chen):
    """Mirror of CHENtoRGB."""
    chen = np.asarray(chen, dtype=np.float64)
    h = chen[..., 0] * 6.283185307179586
    c = chen[..., 1] * 0.9553166181245093
    length = chen[..., 2] * _SQRT3

    ctr = length * np.sin(c) * np.cos(h)
    ctg = length * np.sin(c) * np.sin(h)
    ctb = length * np.cos(c)

    r = ctr * 0.81649658 + ctb * 0.57735027
    g = ctr * -0.40824829 + ctg * 0.70710678 + ctb * 0.57735027
    b = ctr * -0.40824829 + ctg * -0.70710678 + ctb * 0.57735027
    return np.stack([r, g, b], axis=-1)


def _mix_sat(contrasted, toned, mix, contrast, pivot, contrast_fn):
    """Mirror of the mix_sat family: re-apply the SAME contrast to the
    CHEN luma channel only, then blend contrasted -> luma-only by mix.

    contrast == 1 exact-limit guard: every contrast fn is identity
    there, so the luma-only branch IS `toned` — computing it through
    the CHEN round-trip would only add ~4e-9 noise (the DCTL's rotation
    constants are truncated to 8 decimals, so forward/inverse are not
    exactly inverse; invisible under its float32, but it would break
    the stage's exact identity)."""
    if contrast == 1.0:
        luma_only = toned
    else:
        chen = rgb_to_chen(toned)
        chen[..., 2] = contrast_fn(chen[..., 2], contrast, pivot)
        luma_only = chen_to_rgb(chen)
    return contrasted * (1.0 - mix) + luma_only * mix


# ---------------------------------------------------- hi/lo luma masks

def hi_lo_mask_value(metric_rgb, hi_threshold, hi_feather,
                     lo_threshold, lo_feather):
    """Mirror of HiLoMaskValue: 1 at the extreme ends (below the low
    band / above the high band), 0 in the mids, smootherstep feathers
    in log2 stops around mid-grey."""
    m = np.max(np.asarray(metric_rgb, dtype=np.float64), axis=-1)
    mlog = np.log2(np.maximum(m, 1e-10))

    lo_min = np.log2(MID_GREY * 2.0 ** (lo_threshold - 0.5 * lo_feather))
    lo_max = np.log2(MID_GREY * 2.0 ** (lo_threshold + 0.5 * lo_feather))
    hi_min = np.log2(MID_GREY * 2.0 ** (hi_threshold - 0.5 * hi_feather))
    hi_max = np.log2(MID_GREY * 2.0 ** (hi_threshold + 0.5 * hi_feather))

    shadow = 1.0 - smootherstep((mlog - lo_min) / max(lo_max - lo_min, 1e-6))
    highlight = smootherstep((mlog - hi_min) / max(hi_max - hi_min, 1e-6))
    return np.clip(shadow + highlight, 0.0, 1.0)


# -------------------------------------------------------- the pipeline

def filmic_contrast(x: np.ndarray,
                    exposure: float, contrast: float, pivot: float,
                    white_point: float, shoulder: float,
                    shoulder_falloff: float,
                    black_point: float, toe: float, toe_falloff: float,
                    mix_contrast: float, preserve_color: float,
                    pin_ends: float, pop_mids: float,
                    flare: float) -> np.ndarray:
    """The full transform() pixel path (LogC3, Preserve Mid-gray ON,
    linear end exposure). Parameter names follow the DCTL sliders in
    order; `shoulder`/`toe` are the roll pivots, `*_falloff` the
    fall-off sliders."""
    x = np.asarray(x, dtype=np.float64)
    clean = x

    pivot = set_pivot(pivot)

    # --- adaptive shadow/highlight feathering (drives the pin mask)
    pin_n = float(gentle_ramp(pin_ends))
    cdelta = contrast - 1.0
    c_n = float(gentle_ramp(cdelta / 1.0 if cdelta >= 0.0
                            else -cdelta / 0.5))
    feather_drive = float(soft_union(pin_n, c_n))
    hi_drive = feather_drive * feather_drive
    hi_feather = float(np.clip(2.5 + 1.5 * hi_drive, 2.5, 2.5 + 1.5))
    lo_feather = float(np.clip(7.5 + 4.0 * feather_drive, 7.5, 7.5 + 4.0))
    hi_threshold, lo_threshold = 0.8, -3.0

    # --- sanitize black and white points (DCTL order preserved: the
    # shoulder pivot caps against the RAW white point slider)
    w_p_pivot = min(white_point, shoulder)
    shoulder_str = max(10.0 - shoulder_falloff, 0.3)
    toe_str = max(10.0 - toe_falloff, 0.17)

    black_point = 1.0 - black_point * 0.4
    b_p_pivot = min(black_point - 0.05, toe)
    # extended-down White Point: the stock 0.5 floor is removed (the
    # pivot*1.1 limit below + the 1e-3 roll-ratio floor keep it safe)
    white_point = white_point * 0.5 + 0.49
    # stock floor is 0.69 (dead-zones the slider past ~0.775); lowered
    # to 0.4 for the extended 1.5 range (see module docstring)
    black_point = max(black_point, 0.4)

    # --- Preserve Mid-gray (toggle ON, the default)
    pivot = min(0.90, max(0.05, pivot))
    # extended-down White Point: stock hard 0.7 floor removed
    white_point = max(white_point, pivot * 1.1)
    w_p_pivot = max(max(pivot * 1.015, w_p_pivot), w_p_pivot)
    b_p_pivot = max(1.0 - max(min(0.6, 1.0 - b_p_pivot), pivot), pivot)
    if pivot > 0.7:
        b_p_pivot = min(pivot, (1.0 - b_p_pivot) * pivot + 0.5)
    black_point = black_point * 0.45 + 0.55
    # widened toe-shape range (stock 1.48..2.95 -> 0.25..3.35; the
    # default falloff 2 still maps to 2.65 exactly)
    toe_str = max(toe_str * 0.35 - 0.15, 0.25)

    # --- flip Preserve Color for negative contrast (DCTL intuition fix)
    if contrast < 1.0:
        preserve_color = 1.0 - preserve_color

    mix_n = float(np.clip(mix_contrast, 0.0, 1.0))
    w_lin = gauss_weight(mix_n, 0.00, 0.07)
    w_roll = gauss_weight(mix_n, 0.33, 0.18)
    w_pow = gauss_weight(mix_n, 1.00, 0.32)
    w_sum = max(w_lin + w_roll + w_pow, 1e-8)

    def _tone_block(v):
        """White/black rolls + the 3-way contrast blend (shared by the
        image and the mid-grey reference pixel for the Pop Mids anchor)."""
        toned = apply_white_point(v, white_point, w_p_pivot, shoulder_str)
        toned = apply_black_point(toned, black_point, b_p_pivot, toe_str)
        lin = apply_linear_contrast(toned, contrast, pivot)
        lin = _mix_sat(lin, toned, preserve_color, contrast, pivot,
                       apply_linear_contrast)
        roll = apply_rolling_contrast(toned, contrast, pivot)
        roll = _mix_sat(roll, toned, preserve_color, contrast, pivot,
                        apply_rolling_contrast)
        powsig = apply_power_sigmoid_contrast(toned, contrast, pivot)
        powsig = _mix_sat(powsig, toned, preserve_color, contrast, pivot,
                          apply_power_sigmoid_contrast)
        return (lin * (w_lin / w_sum) + roll * (w_roll / w_sum)
                + powsig * (w_pow / w_sum))

    # --- white / black point compression + contrast
    out = _tone_block(x)

    # --- pop mids: exposure bite in the band between two hi/lo masks,
    # measured on the CLEAN input
    if pop_mids != 0.0:
        big = hi_lo_mask_value(clean, 1.04, 1.85,
                               lo_threshold + 1.61, lo_feather - 4.4)
        small = hi_lo_mask_value(clean, 1.02, 1.51, -2.9, 4.5)
        band = np.clip(big - small, 0.0, 1.0)[..., None]
        pop = lin_to_logc3(logc3_to_lin(out) * 2.0 ** -pop_mids)
        out = out * (1.0 - band) + pop * band

        # mid-grey compensation (Marc, 2026-07-22, NOT in the stock
        # tool): the band feather overlaps mid-grey slightly, so Pop
        # Mids alone drifts it. A global counter-gain re-anchors a
        # mid-grey pixel EXACTLY where the tone block put it, so Pop
        # Mids and Exposure are fully decoupled — in the fit and on
        # the desk. Mirrored in our DCTL copy.
        midpix = np.full((1, 3), MID_GREY)
        m0 = float(_tone_block(midpix)[0, 0])
        b_mid = float(np.clip(
            hi_lo_mask_value(midpix, 1.04, 1.85,
                             lo_threshold + 1.61, lo_feather - 4.4)
            - hi_lo_mask_value(midpix, 1.02, 1.51, -2.9, 4.5),
            0.0, 1.0)[0])
        m0_lin = float(logc3_to_lin(m0))
        pop_mid = float(lin_to_logc3(m0_lin * 2.0 ** -pop_mids))
        v_mid = m0 * (1.0 - b_mid) + pop_mid * b_mid
        comp = m0_lin / float(logc3_to_lin(v_mid))
        out = lin_to_logc3(logc3_to_lin(out) * comp)

    # --- exposure: linear gain at the end (exp_toggle == 0, default) —
    # achromatic in linear, moves mid-grey by exactly `exposure` stops
    if exposure != 0.0:
        out = lin_to_logc3(logc3_to_lin(out) * 2.0 ** exposure)

    # --- pin hi/lo ends back to the clean input
    if pin_ends != 0.0:
        mask = hi_lo_mask_value(clean, hi_threshold, hi_feather,
                                lo_threshold, lo_feather)[..., None]
        pinends = out * (1.0 - mask) + clean * mask
        out = out * (1.0 - pin_ends) + pinends * pin_ends

    # --- flare: signed linear offset
    if flare != 0.0:
        offset = remap_asym_signed(flare, -0.01, 0.01)
        out = lin_to_logc3(logc3_to_lin(out) + offset)

    return out


# ------------------------------------------------------------ the stage

from app.core.stage_base import Stage  # noqa: E402  (kept by the module tail)


class FilmicContrastStage(Stage):
    """ME_Filmic Contrast as a fittable stage — Marc's tone node of
    choice for the chain ("this really works"). 14 float sliders, in
    the DCTL's sliderFloatParam0..13 order, values paste 1:1 into the
    (themed) FilmicContrast.dctl:

      Exposure          stops, linear gain applied at the END (mid-grey
                        referenced, achromatic in linear light)
      Contrast          mid slope around the pivot (1 = identity)
      Pivot             relative to mid-grey (0 = mid-grey, LogC3)
      White Point       highlight compression target; 1.02 = OFF (the
                        stock UI tops out at 1.018 ~= the same thing);
                        extended DOWN to -0.15 — the ceiling can fade
                        to just above mid-grey (stock floored at 0.7)
      Shoulder          where the highlight roll starts (code value).
                        NOTE it only shapes anything when White Point
                        is engaged (< 1) — at WP ~1 the roll is
                        invisible no matter where it starts
      Shoulder Falloff  roll softness 0..9.7 (higher = softer knee;
                        extended, stock stopped at 9)
      Black Point       shadow lift depth; 0 = OFF, extended range to
                        1.5 (stock 0.5; sanitize floor lowered, see
                        module docstring)
      Toe               where the shadow roll starts
      Toe Falloff       toe softness 0..10; remapped to a much wider
                        internal range (0.25..3.35 vs stock 1.48..2.95,
                        default unchanged)
      Linear Rolled     blends Linear -> Rolled -> PowerP contrast
      Preserve Color    0 = per-RGB (contrast moves saturation),
                        1 = luma-only via the CHEN model
      Pin Ends          pins extreme shadows/highlights back to the
                        untouched input (adaptive feather)
      Pop Mids          darkens the off-mid band -> mids pop; mid-grey
                        compensated (toolkit addition), so it never
                        moves mid-grey and is decoupled from Exposure
      Flare             signed milky-shadow linear offset
    """

    name = "Filmic Contrast"
    param_names = [
        "Exposure", "Contrast", "Pivot",
        "White Point", "Shoulder", "Shoulder Falloff",
        "Black Point", "Toe", "Toe Falloff",
        "Linear Rolled", "Preserve Color", "Pin Ends", "Pop Mids",
        "Flare",
    ]

    def identity(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0,
                         1.02, 0.6, 6.0,
                         0.0, 0.5, 2.0,
                         0.0, 0.5, 0.0, 0.0,
                         0.0])

    def init(self) -> np.ndarray:
        """Start with the white/black points ENGAGED (the DCTL's own
        defaults): at identity both sections sit in their passthrough
        branch — a dead-gradient region the solver could never leave."""
        p = self.identity()
        p[3] = 1.015   # White Point: the stock default, gentle shoulder
        p[6] = 0.05    # Black Point: just engaged, toe has gradient
        return p

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([-4.0, 0.5, -1.0,
                       -0.15, 0.2, 0.0,
                       0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, -1.0,
                       -1.0])
        hi = np.array([4.0, 3.0, 1.0,
                       1.02, 0.997, 9.7,
                       1.5, 0.8, 10.0,
                       1.0, 1.0, 1.0, 3.5,
                       1.0])
        return lo, hi

    def apply(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return filmic_contrast(x, *[float(v) for v in params])

    def label(self, params: np.ndarray) -> str:
        (exposure, contrast, _pivot, white_point, _sh, _shf,
         black_point, _toe, _toef, _mix, preserve, pin, pop,
         flare) = [float(v) for v in params]
        parts = []
        if contrast > 1.02:
            parts.append("punch contrast")
        elif contrast < 0.98:
            parts.append("flatten")
        if white_point < 1.0:
            parts.append("roll highs")
        if black_point > 0.02:
            parts.append("lift blacks")
        if flare > 0.05:
            parts.append("flare")
        elif flare < -0.05:
            parts.append("de-flare")
        if abs(exposure) > 0.05:
            parts.append(f"{exposure:+.1f} stop")
        if pop > 0.1:
            parts.append("pop mids")
        if pin > 0.3:
            parts.append("pin ends")
        if not parts:
            return "(idle)"
        note = ", ".join(parts)
        if contrast > 1.02 and preserve < 0.3:
            note += " (rich)"
        return note
