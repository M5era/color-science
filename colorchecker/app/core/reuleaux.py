"""1:1 Python port of Reuleaux (hotgluebanjo & calvinsilly).

Source: https://github.com/hotgluebanjo/reuleaux —
resolve/Reuleaux.dctl and extra/ReuleauxUserStandalone.dctl,
transcribed formula-for-formula (vectorized, float64). The upstream
repo carries NO license file: this port exists for private evaluation
of the model (proof of concept) and must not be redistributed.

Every function mirrors its DCTL counterpart exactly, including edge
behavior (sat guard at rot.z == 0, curve endpoint clamping, the
1/sat_factor forward convention, EPS floor on the value factor).
"""

from dataclasses import dataclass, field

import numpy as np

_PI = 3.141592653589  # PI_LOCAL in the DCTL, not numpy's pi
_NORM = np.array([2.0 * _PI, np.sqrt(2.0), 1.0])
_EPS = 1e-6


def rgb_to_reuleaux(rgb: np.ndarray) -> np.ndarray:
    """Mirror of rgb_to_reuleaux: RGB -> (hue, sat, val), all in ~[0,1]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    rot_x = np.sqrt(2.0) / 6.0 * (2.0 * r - g - b)
    rot_y = (g - b) / np.sqrt(6.0)
    rot_z = (r + g + b) / 3.0

    hue = _PI - np.arctan2(rot_y, -rot_x)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(rot_z == 0.0, 0.0, np.hypot(rot_x, rot_y) / rot_z)
    val = np.maximum(r, np.maximum(g, b))

    return np.stack([hue, sat, val], axis=-1) / _NORM


def reuleaux_to_rgb(reuleaux: np.ndarray) -> np.ndarray:
    """Mirror of reuleaux_to_rgb."""
    reuleaux = np.asarray(reuleaux, dtype=np.float64) * _NORM
    hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        m = _NORM[1] * np.maximum.reduce([
            np.cos(hue),
            np.cos(hue + _NORM[0] / 3.0),
            np.cos(hue - _NORM[0] / 3.0),
        ]) + 1.0 / sat

        ocs_x = val * np.cos(hue) / m
        ocs_y = val * np.sin(hue) / m
    ocs_z = val
    # sat == 0 -> m == inf -> ocs_x = ocs_y = 0 (neutral axis), like the GPU.
    ocs_x = np.nan_to_num(ocs_x, nan=0.0, posinf=0.0, neginf=0.0)
    ocs_y = np.nan_to_num(ocs_y, nan=0.0, posinf=0.0, neginf=0.0)

    s32 = np.sqrt(3.0 / 2.0)
    s3 = np.sqrt(3.0)
    r = ocs_z - s32 * np.maximum(np.abs(ocs_y) - s3 * ocs_x, 0.0)
    g = ocs_z - s32 * (np.maximum(np.abs(ocs_y), s3 * ocs_x) - ocs_y)
    b = ocs_z - s32 * (np.maximum(np.abs(ocs_y), s3 * ocs_x) + ocs_y)
    return np.stack([r, g, b], axis=-1)


def _spow(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Sign-preserving power, as in the DCTL."""
    return np.sign(x) * np.abs(x) ** p


def _interp_linear(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mirror of interp_linear: piecewise linear, clamped at the ends.
    (np.interp has identical behavior for ascending xs, which the fixed
    hue anchors always are in the forward direction.)"""
    return np.interp(x, xs, ys)


@dataclass
class ReuleauxUserParams:
    """The ReuleauxUserStandalone sliders, DCTL defaults."""

    overall_sat: float = 1.0
    overall_val: float = 0.0
    # per vector: (hue, sat, val) — DCTL ranges: hue ±0.166, sat 0..2, val ±3
    red: tuple = (0.0, 1.0, 0.0)
    yellow: tuple = (0.0, 1.0, 0.0)
    green: tuple = (0.0, 1.0, 0.0)
    cyan: tuple = (0.0, 1.0, 0.0)
    blue: tuple = (0.0, 1.0, 0.0)
    magenta: tuple = (0.0, 1.0, 0.0)


def reuleaux_user(rgb: np.ndarray, params: ReuleauxUserParams,
                  invert: bool = False) -> np.ndarray:
    """Mirror of ReuleauxUserStandalone's transform()."""
    p = params
    reuleaux = rgb_to_reuleaux(rgb)
    hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

    # 6 hue anchors, 1 wrap below, 2 above — exactly the DCTL's 9 points.
    anchors = np.array([5/6 - 1, 0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0, 1/6 + 1])
    hue_offsets = np.array([p.magenta[0], p.red[0], p.yellow[0], p.green[0],
                            p.cyan[0], p.blue[0], p.magenta[0], p.red[0], p.yellow[0]])
    hue_ys = anchors + hue_offsets
    sat_ys = np.array([p.magenta[1], p.red[1], p.yellow[1], p.green[1],
                       p.cyan[1], p.blue[1], p.magenta[1], p.red[1], p.yellow[1]])
    val_ys = np.array([p.magenta[2], p.red[2], p.yellow[2], p.green[2],
                       p.cyan[2], p.blue[2], p.magenta[2], p.red[2], p.yellow[2]])

    if invert:
        hue_result = _interp_linear(hue_ys, anchors, hue)  # swapped points
    else:
        hue_result = _interp_linear(anchors, hue_ys, hue)
    hue_switch = hue if invert else hue_result

    sat_factor = _interp_linear(anchors, sat_ys, hue_switch) * p.overall_sat
    val_factor = _interp_linear(anchors, val_ys, hue_switch) + p.overall_val

    if not invert:
        with np.errstate(divide="ignore"):
            sat_factor = 1.0 / sat_factor

    sat_result = _spow(sat, sat_factor)
    sat_switch = sat if invert else sat_result

    val_factor = np.maximum(1.0 + sat_switch * val_factor, _EPS)
    if invert:
        val_result = val / val_factor
    else:
        val_result = val * val_factor

    return reuleaux_to_rgb(np.stack([hue_result, sat_result, val_result], axis=-1))
