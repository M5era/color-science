"""PyTorch mirrors of the parametric stages (app/core/stages.py).

Only imported when the torch backend is requested — torch is an
OPTIONAL dependency; nothing else in the app touches this module.

Every mirror computes the same function as its numpy stage (float64,
same constants, same edge conventions) but differentiably, so autograd
gives exact gradients where the numpy path uses finite differences.
The only deliberate deviations are gradient guards at measure-zero
points (exact neutrals, exact zeros) where the true derivative is
undefined; they perturb values by < 1e-12.

Parity with the numpy stages is enforced by tests/test_backprop.py.
"""

import numpy as np
import torch

from app.core import chromogen
from app.core.chromogen import (
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
    SplitToneStage,
)
from app.core.stages import (
    LiftGammaGainStage,
    LinearMatrixStage,
    LumaCurveStage,
    ReuleauxBroadStage,
    ReuleauxFineStage,
    RGBCurvesStage,
)

_PI = 3.141592653589  # PI_LOCAL, as in app/core/reuleaux.py
_SQRT2 = float(np.sqrt(2.0))
_EPS = 1e-6


# ------------------------------------------------------- reuleaux space

def _rgb_to_reuleaux(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rot_x = _SQRT2 / 6.0 * (2.0 * r - g - b)
    rot_y = (g - b) / float(np.sqrt(6.0))
    rot_z = (r + g + b) / 3.0

    hue = _PI - torch.atan2(rot_y, -rot_x)
    # grad-safe hypot (sqrt'(0) is inf); the 1e-24 shifts values < 1e-12
    rad = torch.sqrt(rot_x * rot_x + rot_y * rot_y + 1e-24)
    safe_z = torch.where(rot_z == 0.0, torch.ones_like(rot_z), rot_z)
    sat = torch.where(rot_z == 0.0, torch.zeros_like(rot_z), rad / safe_z)
    val = torch.maximum(r, torch.maximum(g, b))
    return hue / (2.0 * _PI), sat / _SQRT2, val


def _reuleaux_to_rgb(hue, sat, val):
    hue = hue * (2.0 * _PI)
    sat = sat * _SQRT2

    m_cos = _SQRT2 * torch.maximum(
        torch.cos(hue),
        torch.maximum(torch.cos(hue + 2.0 * _PI / 3.0),
                      torch.cos(hue - 2.0 * _PI / 3.0)),
    )
    # numpy: sat == 0 -> 1/sat == inf -> ocs 0; tiny stand-in keeps
    # autograd finite and lands within 1e-12 of that
    sat_safe = torch.where(sat == 0.0, torch.full_like(sat, 1e-20), sat)
    m = m_cos + 1.0 / sat_safe

    ocs_x = val * torch.cos(hue) / m
    ocs_y = val * torch.sin(hue) / m
    ocs_z = val

    s32 = float(np.sqrt(1.5))
    s3 = float(np.sqrt(3.0))
    r = ocs_z - s32 * torch.clamp(torch.abs(ocs_y) - s3 * ocs_x, min=0.0)
    g = ocs_z - s32 * (torch.maximum(torch.abs(ocs_y), s3 * ocs_x) - ocs_y)
    b = ocs_z - s32 * (torch.maximum(torch.abs(ocs_y), s3 * ocs_x) + ocs_y)
    return torch.stack([r, g, b], dim=-1)


def _spow(x, p):
    # sign-preserving power; |x| floored so d/dp (via log|x|) stays finite
    return torch.sign(x) * torch.abs(x).clamp(min=1e-12) ** p


def _interp(x, xs, ys, extrapolate):
    """Differentiable piecewise-linear lookup. `xs` fixed ascending;
    gradients flow through `x` AND `ys`. extrapolate=False clamps at
    the end knots (np.interp), True extends the end segments
    (stages._interp_extrap)."""
    idx = torch.searchsorted(xs, x.detach().reshape(-1).contiguous())
    idx = idx.clamp(1, xs.numel() - 1).reshape(x.shape)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    t = (x - x0) / (x1 - x0)
    if not extrapolate:
        t = t.clamp(0.0, 1.0)
    return y0 + t * (y1 - y0)


# ------------------------------------------------------ smooth windows

def _shoulder(d, flat, soft):
    t = ((d - flat) / torch.clamp(soft, min=1e-6)).clamp(0.0, 1.0)
    return torch.cos(0.5 * torch.pi * t) ** 2


def _plateau_window(x, center, flat, soft):
    return _shoulder(torch.abs(x - center), flat, soft)


def _wrapped_window(x, center, flat, soft):
    v = x - center + 0.5
    d = torch.abs(v - torch.floor(v) - 0.5)
    return _shoulder(d, flat, soft)


# ------------------------------------------------------- stage mirrors

def _matrix_apply(stage, x, p):
    return x @ p.reshape(3, 3).T


def _lgg_apply(stage, x, p):
    y = p[2:5] * (x + p[0] * (1.0 - x))
    return _spow(y, 1.0 / p[1])


def _curve_ys(block):
    return block[0] + torch.cat(
        [block.new_zeros(1), torch.cumsum(block[1:], dim=0)]
    )


def _luma_apply(stage, x, p):
    xs = torch.as_tensor(stage._curve.xs, dtype=x.dtype)
    return _interp(x, xs, _curve_ys(p), extrapolate=True)


def _rgb_curves_apply(stage, x, p):
    n = stage._curve.n_params
    xs = torch.as_tensor(stage._curve.xs, dtype=x.dtype)
    channels = [
        _interp(x[..., c], xs, _curve_ys(p[c * n : (c + 1) * n]),
                extrapolate=True)
        for c in range(3)
    ]
    return torch.stack(channels, dim=-1)


# fixed 9-point anchor grid and the color order M,R,Y,G,C,B,M,R,Y
_ANCHORS = np.array([5 / 6 - 1, 0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0, 1 / 6 + 1])
_WRAP = [5, 0, 1, 2, 3, 4, 5, 0, 1]
_HUE_IDX = torch.tensor([2 + 3 * c for c in _WRAP])
_SAT_IDX = torch.tensor([3 + 3 * c for c in _WRAP])
_VAL_IDX = torch.tensor([4 + 3 * c for c in _WRAP])


def _broad_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    anchors = torch.as_tensor(_ANCHORS, dtype=x.dtype)

    hue_result = _interp(hue, anchors, anchors + p[_HUE_IDX], extrapolate=False)
    sat_factor = _interp(hue_result, anchors, p[_SAT_IDX], extrapolate=False) * p[0]
    sat_result = _spow(sat, 1.0 / sat_factor)
    val_factor = _interp(hue_result, anchors, p[_VAL_IDX], extrapolate=False) + p[1]
    val_result = val * torch.clamp(1.0 + sat_result * val_factor, min=_EPS)
    return _reuleaux_to_rgb(hue_result, sat_result, val_result)


def _fine_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    center = p[0] - torch.floor(p[0])

    w = (
        _wrapped_window(hue, center, p[1], p[2])
        * _plateau_window(val, p[6], p[7], p[8])
        * _plateau_window(sat, p[9], p[10], p[11])
    )

    hue_result = hue + w * p[3]
    sat_factor = 1.0 + w * (p[4] - 1.0)
    sat_result = _spow(sat, 1.0 / sat_factor)
    val_result = val * torch.clamp(1.0 + sat_result * (w * p[5]), min=1e-6)
    return _reuleaux_to_rgb(hue_result, sat_result, val_result)


# ------------------------------------------- chromogen-family mirrors

def _ramp(x, pivot, falloff):
    t = ((x - pivot) / torch.clamp(falloff, min=1e-6) + 0.5).clamp(0.0, 1.0)
    return torch.sin(0.5 * torch.pi * t) ** 2


def _modulation(val, sat, zone, pivot, chroma):
    r = _ramp(val,
              chromogen.MID_GREY + pivot * chromogen.STOP,
              torch.as_tensor(chromogen.LUMA_FALLOFF, dtype=val.dtype))
    m_luma = 1.0 - torch.abs(zone) + torch.abs(zone) * torch.where(
        zone >= 0.0, r, 1.0 - r
    )
    rs = _ramp(sat,
               torch.as_tensor(chromogen.SAT_GATE_PIVOT, dtype=val.dtype),
               torch.as_tensor(chromogen.SAT_GATE_FALLOFF, dtype=val.dtype))
    m_chroma = 1.0 - torch.abs(chroma) + torch.abs(chroma) * torch.where(
        chroma >= 0.0, rs, 1.0 - rs
    )
    return m_luma * m_chroma


def _to_chroma_vec(hue, sat):
    ang = hue * (2.0 * torch.pi)
    return sat * torch.cos(ang), sat * torch.sin(ang)


def _from_chroma_vec(c1, c2):
    sat = torch.sqrt(c1 * c1 + c2 * c2 + 1e-24)
    hue = torch.atan2(c2, c1) / (2.0 * torch.pi)
    return hue - torch.floor(hue), sat


def _rygb_interp(hue, amounts4):
    xs = torch.as_tensor(chromogen._RYGB_XS, dtype=hue.dtype)
    ys = amounts4[list(chromogen._RYGB_WRAP)]
    return _interp(hue, xs, ys, extrapolate=False)


def _softplus_t(x, width):
    return width * torch.nn.functional.softplus(x / width)


def _colour_saturation_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    m = _modulation(val, sat, p[2], p[3], p[4])
    c1, c2 = _to_chroma_vec(hue, sat)

    theta = torch.as_tensor(chromogen.YB_AXIS_TURNS * 2.0 * np.pi, dtype=x.dtype)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    u = cos_t * c1 + sin_t * c2
    v = -sin_t * c1 + cos_t * c2
    u = u * (1.0 + m * (p[1] - 1.0))
    v = v * (1.0 + m * (p[0] - 1.0))
    c1 = cos_t * u - sin_t * v
    c2 = sin_t * u + cos_t * v
    hue2, sat2 = _from_chroma_vec(c1, c2)
    return _reuleaux_to_rgb(hue2, sat2, val)


def _contrast_curve_scalar(v, p, stage):
    """Differentiable mirror of ContrastCurveStage._tone (float64)."""
    mg = chromogen.MID_GREY
    st = chromogen.STOP
    contrast, bp, wp = p[0], p[1], p[2]
    toe_len, toe_str = p[3], p[4]
    sh_len, sh_str = p[5], p[6]
    mid_push = p[8]
    flare = p[10]
    comp = p[12]

    # exposure (p[11]) is applied achromatically in _contrast_curve_apply,
    # NOT here — a per-channel exposure shift would tint through the curve.
    fw = torch.as_tensor(stage._FLARE_WIDTH * st, dtype=v.dtype)
    shadow_w = 1.0 - _ramp(v, mg, fw)
    v = v + flare * stage._FLARE_SCALE * shadow_w
    s = (v - mg) / st

    def gsc(u, n):
        return u / torch.pow(1.0 + torch.pow(torch.abs(u), n), 1.0 / n)

    y = contrast * s
    yk_hi = (1.0 - sh_len) * wp
    h_hi = wp - yk_hi + stage._EPS
    n_hi = 1.0 + stage._STR_GAIN * sh_str
    e_hi = torch.clamp(y - yk_hi, min=0.0)
    hi = yk_hi + h_hi * gsc(e_hi / h_hi, n_hi)
    yk_lo = (1.0 - toe_len) * bp
    h_lo = bp - yk_lo - stage._EPS
    n_lo = 1.0 + stage._STR_GAIN * toe_str
    e_lo = torch.clamp(y - yk_lo, max=0.0)
    lo = yk_lo + h_lo * gsc(e_lo / h_lo, n_lo)
    curve = torch.where(y > yk_hi, hi, torch.where(y < yk_lo, lo, y))

    u = s / stage._MID_W
    g = torch.exp(-0.5 * u * u)
    shape = (1.0 - comp) * g + comp * (u * float(np.exp(0.5)) * g)
    mid = mid_push * stage._MID_SCALE * shape

    return mg + (curve + mid) * st


def _contrast_curve_apply(stage, x, p):
    preserve, blend = p[7], p[9]
    rgb_out = _contrast_curve_scalar(x, p, stage)
    hue, sat, val = _rgb_to_reuleaux(x)
    luma_out = _reuleaux_to_rgb(hue, sat, _contrast_curve_scalar(val, p, stage))
    curved = (1.0 - preserve) * rgb_out + preserve * luma_out
    # achromatic exposure AFTER the curve: slide the Reuleaux value axis
    # (mid-grey referenced), preserve chroma
    ch, cs, cv = _rgb_to_reuleaux(curved)
    curved = _reuleaux_to_rgb(ch, cs, cv + p[11] * chromogen.STOP)
    return (1.0 - blend) * x + blend * curved


def _highlight_bleach_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    zero = p[6] * 0.0
    w = (
        _rygb_interp(hue, p[:4])
        * _ramp(val, chromogen.MID_GREY + p[4] * chromogen.STOP, p[5] * chromogen.STOP)
        * _modulation(val, sat, zero, zero, p[6])
    )
    return _reuleaux_to_rgb(hue, sat * (1.0 - w), val)


def _neutral_tint_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    r = _ramp(val, chromogen.MID_GREY + p[2] * chromogen.STOP, p[3] * chromogen.STOP)
    side = torch.where(p[1] >= 0.0, r, 1.0 - r)
    zero = p[4] * 0.0
    m = side * _modulation(val, sat, zero, zero, 1.0 - p[4])

    strength = torch.abs(p[1]) * stage.TINT_SCALE
    ang = (p[0] / 360.0) * (2.0 * torch.pi)
    s2, s6, s3 = float(np.sqrt(2.0)), float(np.sqrt(6.0)), float(np.sqrt(3.0))
    d = torch.stack([
        s2 * torch.cos(ang),
        (s6 * torch.sin(ang) - s2 * torch.cos(ang)) / 2.0,
        (-s6 * torch.sin(ang) - s2 * torch.cos(ang)) / 2.0,
    ]) / s3
    return x + (strength * m)[..., None] * d


def _brilliance_reduction_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    w = p[1] * _ramp(sat, p[2], p[3])
    val2 = val * 2.0 ** (-stage.REDUCTION_STOPS * p[0] * w)
    return _reuleaux_to_rgb(hue, sat, val2)


def _colour_crosstalk_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    m = _modulation(val, sat, p[4], p[5], p[6])
    az = torch.abs(p[4])
    inherent = (1.0 - az) * val + az
    c1, c2 = _to_chroma_vec(hue, sat)
    eye = torch.eye(4, dtype=x.dtype)
    for i in range(4):
        w = _rygb_interp(hue, eye[i]) * sat * inherent * m * p[i]
        ang = stage._AXES[i] * 2.0 * np.pi
        c1 = c1 + w * float(np.cos(ang))
        c2 = c2 + w * float(np.sin(ang))
    hue2, sat2 = _from_chroma_vec(c1, c2)
    return _reuleaux_to_rgb(hue2, sat2, val)


def _sector_weight(hue, sat, val, p):
    center = (p[0] / 360.0)
    center = center - torch.floor(center)
    zero_flat = p[2] * 0.0
    return (
        _wrapped_window(hue, center, zero_flat, p[2] / 360.0)
        * _modulation(val, sat, p[3], p[4], p[5])
    )


def _sector_skew_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    w = _sector_weight(hue, sat, val, p)
    return _reuleaux_to_rgb(hue + w * (p[1] / 360.0), sat, val)


def _sector_brightness_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    w = _sector_weight(hue, sat, val, p)
    val2 = val * torch.clamp(1.0 + sat * (w * p[1]), min=1e-6)
    return _reuleaux_to_rgb(hue, sat, val2)


def _sector_saturation_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    w = _sector_weight(hue, sat, val, p)
    sat2 = sat * (1.0 + w * (p[1] - 1.0))
    return _reuleaux_to_rgb(hue, sat2, val)


def _sector_squash_apply(stage, x, p):
    hue, sat, val = _rgb_to_reuleaux(x)
    target = p[0] / 360.0
    target = target - torch.floor(target)
    width = torch.clamp(p[2] / 360.0, min=1e-6)

    v = hue - target + 0.5
    delta = v - torch.floor(v) - 0.5
    t = (torch.abs(delta) / width).clamp(0.0, 1.0)
    w = torch.cos(0.5 * torch.pi * t) ** 2

    m = _modulation(val, sat, p[3], p[4], p[5])
    hue2 = target + delta * (1.0 - (p[1] * m) * w)
    return _reuleaux_to_rgb(hue2, sat, val)


def _split_tone_apply(stage, x, p):
    """Differentiable mirror of SplitToneStage.apply (v3: one cubic
    Bezier per channel over the whole range, C1 linear tails)."""
    third = stage._THIRD
    chans = []
    for ci in range(3):
        y0 = p[0 + ci] * third
        y1 = p[3 + ci] * third
        y2 = 1.0 - (2.0 - p[6 + ci]) * third
        y3 = p[9 + ci]
        v = x[..., ci]
        iv = 1.0 - v
        bez = (y0 * iv ** 3 + 3.0 * y1 * iv ** 2 * v
               + 3.0 * y2 * iv * v * v + y3 * v ** 3)
        lo_ext = y0 + 3.0 * (y1 - y0) * v
        hi_ext = y3 + 3.0 * (y3 - y2) * (v - 1.0)
        chans.append(torch.where(v < 0.0, lo_ext,
                                 torch.where(v > 1.0, hi_ext, bez)))
    return torch.stack(chans, dim=-1)


_APPLY = {
    LiftGammaGainStage: _lgg_apply,
    LinearMatrixStage: _matrix_apply,
    LumaCurveStage: _luma_apply,
    RGBCurvesStage: _rgb_curves_apply,
    ReuleauxBroadStage: _broad_apply,
    ReuleauxFineStage: _fine_apply,
    ColourSaturationStage: _colour_saturation_apply,
    ContrastCurveStage: _contrast_curve_apply,
    HighlightBleachStage: _highlight_bleach_apply,
    NeutralTintStage: _neutral_tint_apply,
    BrillianceReductionStage: _brilliance_reduction_apply,
    ColourCrosstalkStage: _colour_crosstalk_apply,
    SectorSkewStage: _sector_skew_apply,
    SectorBrightnessStage: _sector_brightness_apply,
    SectorSaturationStage: _sector_saturation_apply,
    SectorSquashStage: _sector_squash_apply,
    SplitToneStage: _split_tone_apply,
}


def torch_apply(stage, x, params):
    """Apply `stage` differentiably. x: (N, 3) tensor; params: 1-D tensor."""
    try:
        fn = _APPLY[type(stage)]
    except KeyError:
        raise TypeError(
            f"No torch mirror for stage type {type(stage).__name__} — "
            "add one to app/core/torch_stages.py or use the scipy backend"
        ) from None
    return fn(stage, x, params)


def torch_chain(stages, x, param_list):
    out = x
    for stage, p in zip(stages, param_list):
        out = torch_apply(stage, out, p)
    return out


# -------------------------------------------------- ME Filmic Contrast

def _f_logc3_encode(x):
    from app.core import filmic as F
    arg = torch.clamp(F._A * x + F._B, min=1e-30)   # discarded branch guard
    return torch.where(x > F._CUT,
                       F._C * torch.log10(arg) + F._D,
                       F._E * x + F._F)


def _f_logc3_decode(x):
    from app.core import filmic as F
    return torch.where(x > F._E * F._CUT + F._F,
                       (10.0 ** ((x - F._D) / F._C) - F._B) / F._A,
                       (x - F._F) / F._E)


def _f_smootherstep(x):
    x = torch.clamp(x, 0.0, 1.0)
    return 3.0 * x ** 2 - 2.0 * x ** 3


def _f_smootherstep5(x):
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _f_gentle_ramp(x):
    return _f_smootherstep(torch.clamp(x, 0.0, 1.0)) ** 3


def _f_powerf(base, exp):
    return torch.sign(base) * torch.abs(base).clamp(min=1e-12) ** exp


def _f_linear_contrast(x, contrast, pivot):
    return (x - pivot) * contrast + pivot


def _f_rolling_contrast(x, contrast, pivot):
    p = torch.clamp(pivot, 1e-6, 1.0 - 1e-6)
    e = torch.clamp(0.01 * torch.minimum(p, 1.0 - p), min=1e-4)
    x0, x1 = p - e, p + e
    f_left = _f_powerf(x / p, contrast) * p
    pm = 1.0 - p
    f_right = 1.0 - _f_powerf((1.0 - x) / pm, contrast) * pm
    t = _f_smootherstep5((x - x0) / torch.clamp(x1 - x0, min=1e-6))
    blend = f_left * (1.0 - t) + f_right * t
    return torch.where(x <= x0, f_left,
                       torch.where(x >= x1, f_right, blend))


def _f_power_sigmoid_contrast(x, contrast, pivot):
    t0 = torch.clamp(pivot, 1e-4, 1.0 - 1e-4)
    s0 = torch.clamp(contrast, 0.5, 2.5)
    s0v = float(s0.detach()) if torch.is_tensor(s0) else float(s0)
    if s0v > 1.0:
        s0 = s0 * (1.0 + 0.12 * (s0 - 1.0) / 1.5)
        s0v = float(s0.detach())
    if s0v < 1.0:
        x = torch.clamp(x, 0.0, 1.0)
    if abs(s0v - 1.0) < 1e-5:
        return x
    eps = 1e-6
    denom = s0 - 1.0
    denom = torch.sign(denom) * torch.clamp(torch.abs(denom), min=eps)
    s1 = s0 * (1.0 - t0) / denom
    s2 = s0 * (t0 - 0.0) / denom
    s1 = torch.sign(s1) * torch.clamp(torch.abs(s1), min=eps)
    s2 = torch.sign(s2) * torch.clamp(torch.abs(s2), min=eps)
    d_hi = x - t0
    hi = s0 * d_hi / (d_hi * s0 / s1 + 1.0) + t0
    d_lo = t0 - x
    lo = -s0 * d_lo / (d_lo * s0 / s2 + 1.0) + t0
    return torch.where(x >= t0, hi, lo)


def _f_end_roll(v, point, pivot, strength):
    # overflow-free forms (see app/core/filmic._end_roll): knee
    # strength is unbounded, gradients stay finite
    base = torch.clamp((point - pivot) / (1.0 - pivot), min=1e-3)
    bn = base ** strength
    scale = (1.0 - pivot) * base / torch.clamp(
        1.0 - bn, min=1e-30) ** (1.0 / strength)
    mask = v > pivot
    d = torch.clamp(v - pivot, min=0.0) / scale
    d_lo = torch.clamp(d, max=1.0)
    d_hi = torch.clamp(d, min=1.0)
    denom = torch.where(
        d <= 1.0,
        (1.0 + d_lo ** strength) ** (1.0 / strength),
        d_hi * (1.0 + d_hi ** -strength) ** (1.0 / strength))
    rolled = pivot + scale * d / denom
    return torch.where(mask, rolled, v)


def _f_rgb_to_chen(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rtr = r * 0.81649658 + g * -0.40824829 + b * -0.40824829
    rtg = g * 0.70710678 + b * -0.70710678
    rtb = r * 0.57735027 + g * 0.57735027 + b * 0.57735027
    art = torch.atan2(rtg, rtr)
    sphr = torch.sqrt(rtr * rtr + rtg * rtg + rtb * rtb + 1e-24)
    spht = torch.where(art < 0.0, art + 2.0 * 3.141592653589, art)
    sphp = torch.atan2(torch.sqrt(rtr * rtr + rtg * rtg + 1e-24), rtb)
    return torch.stack([spht * 0.15915494309189535,
                        sphp * 1.0467733744265997,
                        sphr * 0.5773502691896258], dim=-1)


def _f_chen_to_rgb(chen):
    h = chen[..., 0] * 6.283185307179586
    c = chen[..., 1] * 0.9553166181245093
    length = chen[..., 2] * 1.7320508075688772
    ctr = length * torch.sin(c) * torch.cos(h)
    ctg = length * torch.sin(c) * torch.sin(h)
    ctb = length * torch.cos(c)
    r = ctr * 0.81649658 + ctb * 0.57735027
    g = ctr * -0.40824829 + ctg * 0.70710678 + ctb * 0.57735027
    b = ctr * -0.40824829 + ctg * -0.70710678 + ctb * 0.57735027
    return torch.stack([r, g, b], dim=-1)


def _f_mix_sat(contrasted, toned, mix, contrast, pivot, contrast_fn):
    cv = float(contrast.detach()) if torch.is_tensor(contrast) else float(contrast)
    if cv == 1.0:                   # exact limit, as in the numpy port
        luma_only = toned
    else:
        chen = _f_rgb_to_chen(toned)
        l2 = contrast_fn(chen[..., 2], contrast, pivot)
        luma_only = _f_chen_to_rgb(
            torch.stack([chen[..., 0], chen[..., 1], l2], dim=-1))
    return contrasted * (1.0 - mix) + luma_only * mix


def _f_hi_lo_mask(metric_rgb, hi_t, hi_f, lo_t, lo_f, mid_grey_log2):
    m = torch.max(metric_rgb, dim=-1).values
    hi_t, hi_f, lo_t, lo_f = (torch.as_tensor(v, dtype=m.dtype)
                              for v in (hi_t, hi_f, lo_t, lo_f))
    mlog = torch.log2(torch.clamp(m, min=1e-10))
    lo_min = mid_grey_log2 + (lo_t - 0.5 * lo_f)
    lo_max = mid_grey_log2 + (lo_t + 0.5 * lo_f)
    hi_min = mid_grey_log2 + (hi_t - 0.5 * hi_f)
    hi_max = mid_grey_log2 + (hi_t + 0.5 * hi_f)
    shadow = 1.0 - _f_smootherstep(
        (mlog - lo_min) / torch.clamp(lo_max - lo_min, min=1e-6))
    highlight = _f_smootherstep(
        (mlog - hi_min) / torch.clamp(hi_max - hi_min, min=1e-6))
    return torch.clamp(shadow + highlight, 0.0, 1.0)


def _filmic_contrast_apply(stage, x, p):
    """Differentiable mirror of app/core/filmic.filmic_contrast. Python
    branches follow the numpy port's guards exactly (they branch on the
    concrete param values; gradients flow through the taken branch)."""
    from app.core import filmic as F

    (exposure, contrast, pivot_rel, white_point, shoulder,
     shoulder_falloff, black_point, toe, toe_falloff, mix_contrast,
     preserve_color, pin_ends, pop_mids, flare) = [p[i] for i in range(14)]

    def _v(t):          # concrete value for CONTROL FLOW only
        return float(t.detach()) if torch.is_tensor(t) else float(t)

    clean = x

    pivot = torch.clamp(pivot_rel + F.MID_GREY, 0.02, 1.0)

    pin_n = _f_gentle_ramp(pin_ends)
    cdelta = contrast - 1.0
    c_n = _f_gentle_ramp(cdelta / 1.0 if _v(cdelta) >= 0.0
                         else -cdelta / 0.5)
    feather_drive = 1.0 - (1.0 - torch.clamp(pin_n, 0.0, 1.0)) * (
        1.0 - torch.clamp(c_n, 0.0, 1.0))
    hi_drive = feather_drive * feather_drive
    hi_feather = torch.clamp(2.5 + 1.5 * hi_drive, 2.5, 4.0)
    lo_feather = torch.clamp(7.5 + 4.0 * feather_drive, 7.5, 11.5)

    w_p_pivot = torch.minimum(white_point, shoulder)
    shoulder_str = torch.clamp(10.0 - shoulder_falloff, min=0.3)
    toe_str = torch.clamp(10.0 - toe_falloff, min=0.17)

    black_point = 1.0 - black_point * 0.4
    b_p_pivot = torch.minimum(black_point - 0.05, toe)
    white_point = white_point * 0.5 + 0.49   # extended-down, no 0.5 floor
    black_point = torch.clamp(black_point, min=0.4)   # extended range floor

    pivot = torch.clamp(pivot, 0.05, 0.90)
    white_point = torch.maximum(white_point, pivot * 1.1)  # no 0.7 floor
    w_p_pivot = torch.maximum(pivot * 1.015, w_p_pivot)
    b_p_pivot = torch.maximum(
        1.0 - torch.maximum(torch.clamp(1.0 - b_p_pivot, max=0.6), pivot),
        pivot)
    if _v(pivot) > 0.7:
        b_p_pivot = torch.minimum(pivot, (1.0 - b_p_pivot) * pivot + 0.5)
    black_point = black_point * 0.45 + 0.55
    toe_str = torch.clamp(toe_str * 0.35 - 0.15, min=0.25)  # widened range
    if _v(toe_falloff) < 0.0:
        toe_str = 3.35 - toe_falloff       # 1:1 negative side, cont. at 0

    if _v(contrast) < 1.0:
        preserve_color = 1.0 - preserve_color

    mix_n = torch.clamp(mix_contrast, 0.0, 1.0)
    w_lin = torch.exp(-0.5 * ((mix_n - 0.00) / 0.07) ** 2)
    w_roll = torch.exp(-0.5 * ((mix_n - 0.33) / 0.18) ** 2)
    w_pow = torch.exp(-0.5 * ((mix_n - 1.00) / 0.32) ** 2)
    w_sum = torch.clamp(w_lin + w_roll + w_pow, min=1e-8)

    def _tone_block(v):
        if _v(white_point) >= 1.0:
            toned = v
        else:
            toned = _f_end_roll(v, white_point, w_p_pivot, shoulder_str)
        if _v(black_point) < 1.0:
            toned = 1.0 - _f_end_roll(1.0 - toned, black_point, b_p_pivot,
                                      toe_str)
        lin = _f_linear_contrast(toned, contrast, pivot)
        lin = _f_mix_sat(lin, toned, preserve_color, contrast, pivot,
                         _f_linear_contrast)
        roll = _f_rolling_contrast(toned, contrast, pivot)
        roll = _f_mix_sat(roll, toned, preserve_color, contrast, pivot,
                          _f_rolling_contrast)
        powsig = _f_power_sigmoid_contrast(toned, contrast, pivot)
        powsig = _f_mix_sat(powsig, toned, preserve_color, contrast, pivot,
                            _f_power_sigmoid_contrast)
        return (lin * (w_lin / w_sum) + roll * (w_roll / w_sum)
                + powsig * (w_pow / w_sum))

    out = _tone_block(x)

    mid_log2 = float(np.log2(F.MID_GREY))
    if _v(pop_mids) != 0.0:
        big = _f_hi_lo_mask(clean, 1.04, 1.85,
                            -3.0 + 1.61, lo_feather - 4.4, mid_log2)
        small = _f_hi_lo_mask(clean, 1.02, 1.51, -2.9, 4.5, mid_log2)
        band = torch.clamp(big - small, 0.0, 1.0)[..., None]
        pop = _f_logc3_encode(_f_logc3_decode(out) * 2.0 ** -pop_mids)
        out = out * (1.0 - band) + pop * band

        # mid-grey compensation (mirrors app/core/filmic.py)
        midpix = torch.full((1, 3), F.MID_GREY, dtype=x.dtype)
        m0 = _tone_block(midpix)[0, 0]
        b_mid = torch.clamp(
            _f_hi_lo_mask(midpix, 1.04, 1.85,
                          -3.0 + 1.61, lo_feather - 4.4, mid_log2)
            - _f_hi_lo_mask(midpix, 1.02, 1.51, -2.9, 4.5, mid_log2),
            0.0, 1.0)[0]
        m0_lin = _f_logc3_decode(m0)
        pop_mid = _f_logc3_encode(m0_lin * 2.0 ** -pop_mids)
        v_mid = m0 * (1.0 - b_mid) + pop_mid * b_mid
        comp = m0_lin / _f_logc3_decode(v_mid)
        out = _f_logc3_encode(_f_logc3_decode(out) * comp)

    if _v(exposure) != 0.0:
        out = _f_logc3_encode(_f_logc3_decode(out) * 2.0 ** exposure)

    if _v(pin_ends) != 0.0:
        mask = _f_hi_lo_mask(clean, 0.8, hi_feather,
                             -3.0, lo_feather, mid_log2)[..., None]
        pinends = out * (1.0 - mask) + clean * mask
        out = out * (1.0 - pin_ends) + pinends * pin_ends

    if _v(flare) != 0.0:
        u = torch.clamp(flare, -1.0, 1.0)
        offset = u * torch.abs(u) * 0.01
        out = _f_logc3_encode(_f_logc3_decode(out) + offset)

    return out


def _exposure_apply(stage, x, p):
    if float(p[0].detach() if torch.is_tensor(p[0]) else p[0]) == 0.0:
        return x
    return _f_logc3_encode(_f_logc3_decode(x) * 2.0 ** p[0])


from app.core.filmic import ExposureStage, FilmicContrastStage  # noqa: E402

_APPLY[FilmicContrastStage] = _filmic_contrast_apply
_APPLY[ExposureStage] = _exposure_apply
