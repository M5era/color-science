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
    contrast, white, black = p[0], p[1], p[2]
    mid_push = p[3]
    sh_roll, toe_roll = p[4], p[5]
    flare = p[8]
    comp = p[10]

    # exposure (p[9]) is applied achromatically in _contrast_curve_apply,
    # NOT here — a per-channel exposure shift would tint through the curve.
    fw = torch.as_tensor(stage._FLARE_WIDTH * st, dtype=v.dtype)
    shadow_w = 1.0 - _ramp(v, mg, fw)
    v = v + flare * stage._FLARE_SCALE * shadow_w
    s = (v - mg) / st

    a = stage._BASE / (contrast - 1.0 + stage._EPS)
    a_hi = white * a
    a_lo = black * a
    n_hi = stage._KNEE0 - stage._KNEE_SLOPE * sh_roll
    n_lo = stage._KNEE0 - stage._KNEE_SLOPE * toe_roll

    def gsc(u, n):
        return u / torch.pow(1.0 + torch.pow(torch.abs(u), n), 1.0 / n)

    up = a_hi * gsc(contrast * s / a_hi, n_hi)
    dn = a_lo * gsc(contrast * s / a_lo, n_lo)
    curve = torch.where(s >= 0.0, up, dn)

    u = s / stage._MID_W
    g = torch.exp(-0.5 * u * u)
    shape = (1.0 - comp) * g + comp * (u * float(np.exp(0.5)) * g)
    mid = mid_push * stage._MID_SCALE * shape

    return mg + (curve + mid) * st


def _contrast_curve_apply(stage, x, p):
    luma_blend, blend = p[6], p[7]
    # achromatic exposure: slide the Reuleaux value axis, preserve chroma
    he, se, ve = _rgb_to_reuleaux(x)
    xe = _reuleaux_to_rgb(he, se, ve + p[9] * chromogen.STOP)
    rgb_out = _contrast_curve_scalar(xe, p, stage)
    hue, sat, val = _rgb_to_reuleaux(xe)
    luma_out = _reuleaux_to_rgb(hue, sat, _contrast_curve_scalar(val, p, stage))
    curved = (1.0 - luma_blend) * rgb_out + luma_blend * luma_out
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
