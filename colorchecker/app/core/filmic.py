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
    referenced (a standalone Exposure node also exists for the manual
    first-node workflow; Filmic keeps its own for now, Marc);
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
  * Exposure == 0 skips its LogC3<->linear round-trip (a *1.0 gain
    changes nothing mathematically; skipping only avoids float
    round-trip noise so identity is exact).
  * shoulder/toe roll ratios are floored at 1e-3 before the fractional
    power (float32-safe at strength 10; same guard added to our DCTL):
    where the stock tool NaNs (roll pivot beyond the scaled point) the
    limit flattens that end at the pivot instead of breaking the image.
  * Shoulder Falloff extended to 9.7 (reaches the softness floor) and
    the Shoulder slider step fixed 0.1 -> 0.001; Toe Falloff remapped
    in preserve-midgray from strength 1.48..2.95 to 0.25..3.35 (much
    wider toe shapes, the default 2 -> 2.65 unchanged). Marc, 22 eve.
  * Toe/Shoulder POSITION sliders remapped to be LIVE end-to-end
    (2026-07-22 night, Marc: "the actual toe and shoulder sliders dont
    do anything"): the stock clamp chains dead-zoned ~80% of both
    sliders. Full travel now sweeps the valid range linearly —
    Shoulder: just above the pivot -> just under the white ceiling;
    Toe: hugging the black floor -> up to mid-grey.
  * SLIMMED 2026-07-23 (Marc, after fitting all nine test LUTs):
    Linear Rolled, Pin Ends, Pop Mids, Flare and the second toe stage
    removed — fits pinned Linear Rolled at 0 (pure linear contrast)
    and never engaged the second toe; Pin/Pop/Flare never affected
    tone matching. Preserve Color stays. 13 sliders total.
  * Bend Point (the second roll's ceiling) reads in STOPS above mid
    grey (0.5..8.5, >= 8.26 = exactly OFF) instead of the White-Point
    code-linear sanitize — that mapping was dead until ~raw 0.5
    because code is exponential in stops (Marc, 2026-07-23: "only
    starts to kick in around 0.5"). See bend_ceiling().
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


# ------------------------------------------------- contrast primitive
# (Linear Rolled removed 2026-07-23: the all-LUT fits pinned the blend
# at pure LINEAR contrast, so the rolling / power-sigmoid families and
# their gaussian blend are gone.)

def apply_linear_contrast(x, contrast, pivot):
    return (x - pivot) * contrast + pivot


# ------------------------------------------- white / black point rolls

def _end_roll(v, point, pivot, strength):
    """The shared shoulder/toe compressor for values above `pivot`
    (mirror of apply_white_point's per-channel body). `point` is where
    the roll lands at input 1.0. Two ALGEBRAICALLY IDENTICAL but
    overflow-free rewrites (mirrored in the DCTL) so the knee strength
    is unbounded — the naive base**-n and (1+d**n) forms blow past
    float32 around n ~ 12, capping how TIGHT the toe could get (Marc:
    "really really tight"):
      (base^-n - 1)^(1/n)      == (1 - base^n)^(1/n) / base
      (1 + d^n)^(1/n) for d>1  == d * (1 + d^-n)^(1/n)
    Base floored at 1e-3: where the stock tool NaNs (roll pivot beyond
    the scaled point) the limit flattens the end at the pivot."""
    base = max((point - pivot) / (1.0 - pivot), 1e-3)
    bn = base ** strength                      # underflows to 0 = hard knee
    scale = (1.0 - pivot) * base / max(1.0 - bn, 1e-30) ** (1.0 / strength)
    d = np.maximum(v - pivot, 0.0) / scale     # negative side discarded below
    d_lo = np.minimum(d, 1.0)
    d_hi = np.maximum(d, 1.0)
    denom = np.where(
        d <= 1.0,
        (1.0 + d_lo ** strength) ** (1.0 / strength),
        d_hi * (1.0 + d_hi ** -strength) ** (1.0 / strength))
    rolled = pivot + scale * d / denom
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


# ---------------------------------------- per-end sanitize helpers
# (factored out so the SECOND toe/shoulder stage shares them exactly)

def shoulder_strength(falloff: float) -> float:
    """Shoulder Falloff slider -> internal knee strength. Positive is
    the stock linear soft direction (strength 10 -> 0.3); NEGATIVE is
    GEOMETRIC — the knee radius shrinks like 1/strength, so linear
    growth felt dead below 0: strength doubles every 5 slider units
    (10 -> 320 at -25), matching dctl/FilmicContrast.dctl."""
    if falloff < 0.0:
        return 10.0 * 2.0 ** (-falloff / 5.0)
    return max(10.0 - falloff, 0.3)


def toe_strength(falloff: float) -> float:
    """Toe Falloff slider -> internal strength: the widened
    preserve-midgray remap (stock 1.48..2.95 -> 0.25..3.35, default 2
    -> 2.65 unchanged); negative maps 1:1 (continuous at 0 -> 3.35) up
    to near-rectangular (Marc: "almost rectangular... not all the
    way")."""
    if falloff < 0.0:
        return 3.35 - falloff
    return max(max(10.0 - falloff, 0.17) * 0.35 - 0.15, 0.25)


def sanitize_white_point(white_point_raw: float, pivot: float) -> float:
    """Raw White Point slider -> internal ceiling. Extended-down: the
    stock 0.5/0.7 floors are removed (pivot*1.1 + the 1e-3 roll-ratio
    floor keep it safe)."""
    return max(white_point_raw * 0.5 + 0.49, pivot * 1.1)


def bend_ceiling(bend_point_stops: float, pivot: float) -> float:
    """Bend Point slider (in STOPS above mid grey) -> internal ceiling
    code. Replaces the White-Point-style code-linear sanitize for the
    Bend section (Marc 2026-07-23: that mapping was dead until ~0.5 —
    half the travel only shaved invisible speculars, because the
    ceiling moved linearly in CODE which is exponential in stops).
    Uniform slider steps now move the ceiling by uniform exposure.
    >= 8.26 stops puts the ceiling at/above code 1.0 = exactly OFF."""
    ceiling = float(lin_to_logc3(0.18 * 2.0 ** bend_point_stops))
    return max(ceiling, pivot * 1.1)


def sanitize_black_point(black_point_raw: float) -> float:
    """Raw Black Point slider -> internal floor (on the flipped image).
    Stock sanitize floor 0.69 lowered to 0.4 for the extended 1.5
    range."""
    return max(1.0 - black_point_raw * 0.4, 0.4) * 0.45 + 0.55


def shoulder_roll_pivot(shoulder_raw: float, pivot: float,
                        white_point: float) -> float:
    """LIVE Shoulder position mapping (toolkit rewrite): full slider
    travel sweeps just above the pivot .. just under the white
    ceiling."""
    t = min(max((shoulder_raw - 0.2) / (0.997 - 0.2), 0.0), 1.0)
    lo = pivot * 1.015
    hi = max(white_point - 0.01, lo + 1e-4)
    return lo + (hi - lo) * t


def toe_roll_pivot(toe_raw: float, pivot: float,
                   black_point: float) -> float:
    """LIVE Toe position mapping: full travel sweeps hugging the black
    floor (raw 0) .. up to mid-grey (raw 0.8)."""
    t = min(max(toe_raw / 0.8, 0.0), 1.0)
    lo = 1.0 - pivot
    hi = max(black_point - 0.05, lo + 1e-4)
    return lo + (hi - lo) * (1.0 - t)


# -------------------------------------------------------- the pipeline

def filmic_contrast(x: np.ndarray,
                    exposure: float, contrast: float, pivot: float,
                    white_point: float, shoulder: float,
                    shoulder_falloff: float,
                    bend_point: float, bend: float,
                    bend_falloff: float,
                    black_point: float, toe: float, toe_falloff: float,
                    preserve_color: float) -> np.ndarray:
    """The slimmed tone pipeline (LogC3, Preserve Mid-gray ON, linear
    end exposure). Parameter names follow the DCTL sliders in order;
    `shoulder`/`toe` are the roll pivots, `*_falloff` the fall-off
    sliders; the `*2` trio is the SECOND shoulder stage.

    2026-07-23 slimming (Marc, after the all-LUT fits): Linear Rolled,
    Pin Ends, Pop Mids, Flare and the second TOE stage removed — the
    fits pinned Linear Rolled at 0 (pure linear contrast) and never
    engaged the second toe on any of the nine test LUTs; Pin/Pop/Flare
    never affected tone matching. Preserve Color stays."""
    x = np.asarray(x, dtype=np.float64)

    pivot = set_pivot(pivot)

    # --- Preserve Mid-gray (toggle ON, the default)
    pivot = min(0.90, max(0.05, pivot))

    # --- sanitize both shoulder stages + the toe (shared helpers)
    shoulder_str = shoulder_strength(shoulder_falloff)
    shoulder_str2 = shoulder_strength(bend_falloff)
    toe_str = toe_strength(toe_falloff)

    white_point = sanitize_white_point(white_point, pivot)
    bend_ceiling_code = bend_ceiling(bend_point, pivot)
    black_point = sanitize_black_point(black_point)

    w_p_pivot = shoulder_roll_pivot(shoulder, pivot, white_point)
    w_p_pivot2 = shoulder_roll_pivot(bend, pivot, bend_ceiling_code)
    b_p_pivot = toe_roll_pivot(toe, pivot, black_point)

    # --- flip Preserve Color for negative contrast (DCTL intuition fix)
    if contrast < 1.0:
        preserve_color = 1.0 - preserve_color

    # --- white/black rolls + linear contrast (+ Preserve Color mix)
    toned = apply_white_point(x, white_point, w_p_pivot, shoulder_str)
    toned = apply_white_point(toned, bend_ceiling_code, w_p_pivot2,
                              shoulder_str2)
    toned = apply_black_point(toned, black_point, b_p_pivot, toe_str)
    out = apply_linear_contrast(toned, contrast, pivot)
    out = _mix_sat(out, toned, preserve_color, contrast, pivot,
                   apply_linear_contrast)

    # --- exposure: linear gain at the end — achromatic in linear,
    # moves mid-grey by exactly `exposure` stops
    if exposure != 0.0:
        out = lin_to_logc3(logc3_to_lin(out) * 2.0 ** exposure)

    return out


# ------------------------------------------------------------ the stage

from app.core.stage_base import Stage  # noqa: E402  (kept by the module tail)


class FilmicContrastStage(Stage):
    """ME_Filmic Contrast as a fittable stage — Marc's tone node of
    choice for the chain ("this really works"). 13 float sliders, in
    the DCTL's sliderFloatParam0..12 order, values paste 1:1 into the
    (themed) FilmicContrast.dctl.

    SLIMMED 2026-07-23 (Marc, after fitting all nine test LUTs):
    Linear Rolled (fits pinned it at 0 — pure linear contrast),
    Pin Ends, Pop Mids, Flare and the second toe stage removed;
    Preserve Color stays.

      Exposure          stops, linear gain applied at the END (mid-grey
                        referenced, achromatic in linear light)
      Contrast          mid slope around the pivot (1 = identity);
                        ceiling raised 3 -> 5 (the toe holds shadows
                        down, but the steep release above it IS the
                        contrast slope — Marc kept maxing 3.0)
      Pivot             relative to mid-grey (0 = mid-grey, LogC3)
      White Point       highlight compression target; 1.02 = OFF (the
                        stock UI tops out at 1.018 ~= the same thing);
                        extended DOWN to -0.15 — the ceiling can fade
                        to just above mid-grey (stock floored at 0.7)
      Shoulder          where the highlight roll starts (code value).
                        NOTE it only shapes anything when White Point
                        is engaged (< 1) — at WP ~1 the roll is
                        invisible no matter where it starts
      Shoulder Falloff  roll softness -25..9.7: higher = softer knee;
                        NEGATIVE is GEOMETRIC (strength doubles every
                        5 units, 10 -> 320 at -25) so every step
                        visibly tightens toward rectangular — the old
                        linear negative side dead-zoned the bulk of
                        the slider (synced with the DCTL)
      Bend Point /      a SECOND, independent shoulder roll chained
      Bend /            after the first — same math, own landing
      Bend Falloff      point, position and falloff (renamed from
                        "White Point 2 / Shoulder 2" 2026-07-23,
                        Marc: the old names made no sense; "Bend" is
                        his word for the smooth bend of the contrast
                        line it creates). Parked (Bend Point = 1.02 =
                        OFF) by default. Two knees let the shoulder
                        take shapes one roll can't (e.g. a soft early
                        rolloff plus a hard clamp at the top)
      Black Point       shadow lift depth; 0 = OFF, extended range to
                        1.5 (stock 0.5; sanitize floor lowered, see
                        module docstring)
      Toe               where the shadow roll starts
      Toe Falloff       toe softness; extended to -100..10 — negative
                        maps 1:1 to a MUCH sharper toe (strength to
                        ~103, near-rectangular; stock capped at
                        3.35), positive is the stock soft direction,
                        default unchanged
      Preserve Color    0 = per-RGB (contrast moves saturation),
                        1 = luma-only via the CHEN model
    """

    name = "Filmic Contrast"
    param_names = [
        "Exposure", "Contrast", "Pivot",
        "White Point", "Shoulder", "Shoulder Falloff",
        "Bend Point", "Bend", "Bend Falloff",
        "Black Point", "Toe", "Toe Falloff",
        "Preserve Color",
    ]

    def identity(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0,
                         1.02, 0.6, 6.0,
                         8.5, 0.8, 6.0,
                         0.0, 0.5, 2.0,
                         0.5])

    def init(self) -> np.ndarray:
        """Start with the FIRST white/black points ENGAGED (the DCTL's
        own defaults): at identity both sections sit in their
        passthrough branch — a dead-gradient region the solver could
        never leave. The second shoulder stays parked: it is a manual
        shaping tool, and engaging both at init would give the solver
        two collinear knees."""
        p = self.identity()
        p[3] = 1.015   # White Point: the stock default, gentle shoulder
        p[9] = 0.05    # Black Point: just engaged, toe has gradient
        return p

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([-4.0, 0.5, -1.0,
                       -0.15, 0.2, -25.0,
                       0.5, 0.2, -25.0,
                       0.0, 0.0, -100.0,
                       0.0])
        hi = np.array([4.0, 5.0, 1.0,
                       1.02, 0.997, 9.7,
                       8.5, 0.997, 9.7,
                       1.5, 0.8, 10.0,
                       1.0])
        return lo, hi

    def apply(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return filmic_contrast(x, *[float(v) for v in params])

    def label(self, params: np.ndarray) -> str:
        (exposure, contrast, _pivot, white_point, _sh, _shf,
         bend_point, _bend, _bendf,
         black_point, _toe, _toef,
         preserve) = [float(v) for v in params]
        if bend_point < 8.26:            # bend engaged (stops units)
            white_point = min(white_point, 0.99)
        parts = []
        if contrast > 1.02:
            parts.append("punch contrast")
        elif contrast < 0.98:
            parts.append("flatten")
        if white_point < 1.0:
            parts.append("roll highs")
        if black_point > 0.02:
            parts.append("lift blacks")
        if abs(exposure) > 0.05:
            parts.append(f"{exposure:+.1f} stop")
        if not parts:
            return "(idle)"
        note = ", ".join(parts)
        if contrast > 1.02 and preserve < 0.3:
            note += " (rich)"
        return note


class ExposureStage(Stage):
    """Pure linear-gain exposure in stops — split out of Filmic
    Contrast (Marc, 2026-07-22: "have it be its own dctl that lives as
    the very first node, just linear gain"). Achromatic in linear
    light, mid-grey referenced. Not in the ML audition pool: Marc sets
    it manually as node 1; it exists for manual prefixes, presets and
    the .drx export."""

    name = "Exposure"
    param_names = ["Exposure"]

    def identity(self) -> np.ndarray:
        return np.array([0.0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-8.0]), np.array([8.0])

    def apply(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        e = float(params[0])
        x = np.asarray(x, dtype=np.float64)
        if e == 0.0:
            return x.copy()
        return lin_to_logc3(logc3_to_lin(x) * 2.0 ** e)

    def label(self, params: np.ndarray) -> str:
        e = float(params[0])
        if abs(e) < 0.02:
            return "(idle)"
        return f"{e:+.1f} stop"
