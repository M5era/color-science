"""Parametric match stages: pure functions over flat parameter vectors.

The contract every stage obeys — and the reason this is ready for
gradient-based optimization later:

- ALL state is a flat float vector `params` with box `bounds()`
- `apply(x, params)` is pure and vectorized: no hidden state, no
  side effects; swap numpy for torch and the architecture holds
- `identity()` is the do-nothing parameter vector (also the
  regularization anchor, so overlapping stages don't fight)

Stages: LinearMatrixStage (9), LumaCurveStage (monotone shared 1D),
RGBCurvesStage (3 monotone 1D), ReuleauxBroadStage (the validated
fixed-6-anchor port), ReuleauxFineStage (one freely placed hue zone
with smooth hue window + sat mask + luma mask).
"""

import numpy as np

from app.core.reuleaux import (
    ReuleauxUserParams,
    _spow,
    reuleaux_to_rgb,
    reuleaux_user,
    rgb_to_reuleaux,
)
from app.core.stage_base import Stage
from app.core.windows import plateau_window, wrapped_window


class LinearMatrixStage(Stage):
    name = "Matrix"

    def identity(self) -> np.ndarray:
        return np.eye(3).ravel()

    def bounds(self):
        return np.full(9, -2.0), np.full(9, 3.0)

    def apply(self, x, params):
        return x @ params.reshape(3, 3).T

    def describe(self, params):
        rows = params.reshape(3, 3)
        lines = [" ".join(f"{v: .5f}" for v in row) for row in rows]
        return "Matrix (rows R,G,B):\n  " + "\n  ".join(lines)


def _interp_extrap(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Piecewise-linear with linear extrapolation past the end knots
    (a clamped contrast curve would flatten scene values outside 0..1)."""
    y = np.interp(x, xs, ys)
    slope_lo = (ys[1] - ys[0]) / (xs[1] - xs[0])
    slope_hi = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(x < xs[0], ys[0] + (x - xs[0]) * slope_lo, y)
    y = np.where(x > xs[-1], ys[-1] + (x - xs[-1]) * slope_hi, y)
    return y


class _MonotoneCurve:
    """Shared machinery: n knots on a fixed x grid over [0, 1].

    Params per curve: [y0, d1..d_{n-1}] with every delta bounded > 0 —
    the fitted curve is monotone BY CONSTRUCTION, no solve can produce
    tone reversal."""

    def __init__(self, n_points: int):
        self.n_points = n_points
        self.xs = np.linspace(0.0, 1.0, n_points)

    @property
    def n_params(self) -> int:
        return self.n_points

    def identity_curve(self) -> np.ndarray:
        spacing = 1.0 / (self.n_points - 1)
        return np.concatenate([[0.0], np.full(self.n_points - 1, spacing)])

    def curve_bounds(self):
        lo = np.concatenate([[-1.0], np.full(self.n_points - 1, 1e-3)])
        hi = np.concatenate([[1.0], np.full(self.n_points - 1, 1.0)])
        return lo, hi

    def ys(self, params: np.ndarray) -> np.ndarray:
        return params[0] + np.concatenate([[0.0], np.cumsum(params[1:])])

    def evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return _interp_extrap(x, self.xs, self.ys(params))


class LumaCurveStage(Stage):
    """One monotone contrast curve applied identically to R, G and B."""

    name = "Luma Curve"

    def __init__(self, n_points: int = 6):
        self._curve = _MonotoneCurve(n_points)

    def identity(self):
        return self._curve.identity_curve()

    def bounds(self):
        return self._curve.curve_bounds()

    def apply(self, x, params):
        return self._curve.evaluate(x, params)

    def describe(self, params):
        ys = self._curve.ys(params)
        pairs = ", ".join(f"{x:.2f}->{y:.4f}" for x, y in zip(self._curve.xs, ys))
        return f"Luma curve knots: {pairs}"


class RGBCurvesStage(Stage):
    """Three independent monotone curves (split-toning residual)."""

    name = "RGB Curves"

    def __init__(self, n_points: int = 6):
        self._curve = _MonotoneCurve(n_points)

    def identity(self):
        return np.tile(self._curve.identity_curve(), 3)

    def bounds(self):
        lo, hi = self._curve.curve_bounds()
        return np.tile(lo, 3), np.tile(hi, 3)

    def apply(self, x, params):
        n = self._curve.n_params
        out = np.empty_like(x)
        for channel in range(3):
            block = params[channel * n : (channel + 1) * n]
            out[..., channel] = self._curve.evaluate(x[..., channel], block)
        return out

    def describe(self, params):
        n = self._curve.n_params
        parts = []
        for label, channel in (("R", 0), ("G", 1), ("B", 2)):
            ys = self._curve.ys(params[channel * n : (channel + 1) * n])
            pairs = ", ".join(f"{y:.4f}" for y in ys)
            parts.append(f"  {label}: [{pairs}] at x={np.round(self._curve.xs, 2).tolist()}")
        return "RGB curve knots:\n" + "\n".join(parts)


class ReuleauxBroadStage(Stage):
    """The validated 1:1 reuleaux port (6 fixed hue anchors) — the
    broad-strokes stage. Params in DCTL slider order:
    [overall_sat, overall_val, then (hue, sat, val) per R,Y,G,C,B,M]."""

    name = "Reuleaux Broad"

    _COLORS = ("red", "yellow", "green", "cyan", "blue", "magenta")

    def identity(self):
        return np.array([1.0, 0.0] + [0.0, 1.0, 0.0] * 6)

    def bounds(self):
        # DCTL slider ranges; sat floored above 0 (forward path divides by it).
        lo = [0.05, -3.0] + [-0.166, 0.05, -3.0] * 6
        hi = [2.0, 3.0] + [0.166, 2.0, 3.0] * 6
        return np.asarray(lo), np.asarray(hi)

    def _to_params(self, params: np.ndarray) -> ReuleauxUserParams:
        vectors = {
            color: tuple(params[2 + 3 * i : 5 + 3 * i])
            for i, color in enumerate(self._COLORS)
        }
        return ReuleauxUserParams(
            overall_sat=float(params[0]), overall_val=float(params[1]), **vectors
        )

    def apply(self, x, params):
        return reuleaux_user(x, self._to_params(params))

    def describe(self, params):
        lines = [
            f"Reuleaux sliders (paste into ReuleauxUserStandalone.dctl):",
            f"  Overall Saturation: {params[0]:.3f}   Overall Value: {params[1]:.3f}",
        ]
        for i, color in enumerate(self._COLORS):
            h, s, v = params[2 + 3 * i : 5 + 3 * i]
            lines.append(f"  {color.capitalize():8s} Hue {h: .3f}  Sat {s:.3f}  Val {v: .3f}")
        return "\n".join(lines)


class ReuleauxFineStage(Stage):
    """One freely placed hue zone in reuleaux space.

    Unlike the broad stage's 6 fixed anchors, the hue center is a free
    360-degree parameter, and the adjustment is additionally gated by a
    smooth sat mask and luma mask (plateau windows, app.core.windows).
    Chain several Fine stages for several zones.

    Params (12):
      [0] hue_center  (turns; wraps, bounds extend past 0..1 so the
                       solver can cross the red seam continuously)
      [1] hue_flat    full-strength half-width (turns)
      [2] hue_soft    cos^2 falloff width (turns)
      [3] hue_shift   gated delta hue (turns, DCTL-style +-0.166)
      [4] sat_adj     gated sat factor (1 = neutral, DCTL convention)
      [5] val_adj     gated val slider (0 = neutral, scales with sat
                       like the DCTL, so the neutral axis is protected)
      [6..8]  luma mask center/flat/soft over val (identity: wide open)
      [9..11] sat  mask center/flat/soft over sat (identity: wide open)

    All ops at identity => exact identity regardless of the windows,
    and the identity anchor keeps both masks wide open, so the solver
    only narrows a mask when doing so actually pays.
    """

    name = "Reuleaux Fine"
    local_tool = True  # single-hue zone: discounted by the search's broad_bias

    # flat cores that cover the whole working domain == mask off
    _LUMA_OPEN = (0.5, 2.0, 0.25)
    _SAT_OPEN = (0.5, 1.5, 0.25)

    def identity(self):
        return np.array(
            [0.0, 0.04, 0.08, 0.0, 1.0, 0.0]
            + list(self._LUMA_OPEN) + list(self._SAT_OPEN)
        )

    def bounds(self):
        lo = [-0.25, 0.005, 0.01, -0.166, 0.05, -3.0,
              -0.5, 0.0, 0.01,   -0.5, 0.0, 0.01]
        hi = [1.25, 0.5, 0.5, 0.166, 2.0, 3.0,
              2.0, 2.5, 1.0,   1.5, 2.0, 1.0]
        return np.asarray(lo), np.asarray(hi)

    def apply(self, x, params):
        p = np.asarray(params, dtype=np.float64)
        reuleaux = rgb_to_reuleaux(x)
        hue, sat, val = reuleaux[..., 0], reuleaux[..., 1], reuleaux[..., 2]

        w = (
            wrapped_window(hue, p[0] % 1.0, p[1], p[2])
            * plateau_window(val, p[6], p[7], p[8])
            * plateau_window(sat, p[9], p[10], p[11])
        )

        hue_result = hue + w * p[3]
        # gated versions of the DCTL forward ops: blend each factor
        # toward neutral by the mask weight, then apply as the DCTL does
        sat_factor = 1.0 + w * (p[4] - 1.0)
        sat_result = _spow(sat, 1.0 / sat_factor)
        val_result = val * np.maximum(1.0 + sat_result * (w * p[5]), 1e-6)

        return reuleaux_to_rgb(
            np.stack([hue_result, sat_result, val_result], axis=-1)
        )

    def label(self, params):
        from app.core.chromogen import hue_word
        p = np.asarray(params, dtype=np.float64)
        word = hue_word((p[0] % 1.0) * 360.0)
        moves = []
        if abs(p[3]) > 0.008:
            moves.append("shift")
        if abs(p[4] - 1.0) > 0.08:
            moves.append("boost" if p[4] > 1.0 else "desat")
        if abs(p[5]) > 0.08:
            moves.append("brighten" if p[5] > 0 else "darken")
        if not moves:
            return f"zone {word}s (idle)"
        return f"{'/'.join(moves)} {word}s"

    def describe(self, params):
        p = np.asarray(params, dtype=np.float64)
        deg = lambda t: t * 360.0

        def mask_line(label, center, flat, soft, span):
            if flat >= span:
                return f"  {label} mask: wide open (off)"
            return (f"  {label} mask: center {center:.3f}  "
                    f"core ±{flat:.3f}  soft {soft:.3f}")

        return "\n".join([
            "Reuleaux Fine zone (paste into dctl/ReuleauxFine.dctl, degrees):",
            (f"  Hue center {deg(p[0] % 1.0):.1f}°  "
             f"core ±{deg(p[1]):.1f}°  soft {deg(p[2]):.1f}°"),
            (f"  Δhue {deg(p[3]):+.2f}°  Sat ×{p[4]:.3f}  "
             f"Val {p[5]:+.3f}"),
            mask_line("Luma", p[6], p[7], p[8], span=1.5),
            mask_line("Sat", p[9], p[10], p[11], span=1.0),
        ])


class LiftGammaGainStage(Stage):
    """Exposure/white-balance prep: master Lift + master Gamma +
    PER-CHANNEL Gain (ganged gains = exposure, unganged = white
    balance). Smooth and monotone by construction — cannot band, fold
    or clip — so it is safe to put in front of a look chain.

        y_c = gain_c * (x_c + lift * (1 - x_c));  y = spow(y, 1/gamma)

    reg_scale is high on purpose (Marc): the solver assumes exposure/WB
    are already fine and only moves these when it makes the fit a LOT
    easier."""

    name = "Lift Gamma Gain"
    param_names = ["Lift", "Gamma", "Gain R", "Gain G", "Gain B"]
    reg_scale = 25.0

    def identity(self):
        return np.array([0.0, 1.0, 1.0, 1.0, 1.0])

    def bounds(self):
        return (np.asarray([-0.2, 0.6, 0.5, 0.5, 0.5]),
                np.asarray([0.2, 1.6, 2.0, 2.0, 2.0]))

    def apply(self, x, params):
        lift, gamma = params[0], params[1]
        gains = params[2:5]
        y = gains * (x + lift * (1.0 - x))
        return _spow(y, 1.0 / gamma)

    def label(self, params):
        lift, gamma, gr, gg, gb = params
        gains = np.array([gr, gg, gb])
        parts = []
        if gains.max() - gains.min() > 0.05:
            parts.append("white balance")
        if abs(gains.mean() - 1.0) > 0.05 or abs(lift) > 0.03:
            parts.append("exposure trim")
        if abs(gamma - 1.0) > 0.05:
            parts.append("gamma trim")
        return " + ".join(parts) if parts else "prep (idle)"

    def describe(self, params):
        lift, gamma, gr, gg, gb = params
        return "\n".join([
            "Lift Gamma Gain (paste into dctl/LiftGammaGain.dctl):",
            f"  Lift {lift:+.4f}   Gamma {gamma:.4f}   "
            f"Gain R {gr:.4f}  G {gg:.4f}  B {gb:.4f}",
        ])


STAGE_POOL = {
    "Lift Gamma Gain": LiftGammaGainStage,
    "Matrix": LinearMatrixStage,
    "Luma Curve": LumaCurveStage,
    "RGB Curves": RGBCurvesStage,
    "Reuleaux Broad": ReuleauxBroadStage,
    "Reuleaux Fine": ReuleauxFineStage,
}

# Chromogen-style stages live in app/core/chromogen.py; imported at the
# bottom of this module (they subclass Stage) and registered here.

CHAIN_PRESETS = {
    "Full (Luma → RGB → Reuleaux Broad)": ["Luma Curve", "RGB Curves", "Reuleaux Broad"],
    "Reuleaux Broad only": ["Reuleaux Broad"],
    "Matrix + Reuleaux Broad": ["Matrix", "Reuleaux Broad"],
    "Reuleaux Broad + Fine": ["Reuleaux Broad", "Reuleaux Fine"],
    "Chromogen broad (Sat → Crosstalk → Contrast → Bleach → Tint)": [
        "Colour Saturation", "Colour Crosstalk", "Contrast Curve",
        "Highlight Bleach", "Neutral Tint",
    ],
    # the solve MODE for matching: safe prep (strongly anchored at
    # identity) in front of the chromogen look chain
    "Chromogen match (LGG prep → Chromogen chain)": [
        "Lift Gamma Gain", "Colour Saturation", "Colour Crosstalk",
        "Contrast Curve", "Highlight Bleach", "Neutral Tint",
    ],
    # the canonical full stack, in the order of Marc's real Chromogen
    # film look (and Andy's demo): broad first, sector tools BEFORE
    # Highlight Bleach (order matters — sectors want the full scene-
    # referred volume), tints after the bleach, final sat shaping.
    # Duplicate stages are fine; fitted labels tell them apart.
    "Chromogen film look (full stack)": [
        "Lift Gamma Gain",
        "Colour Saturation", "Colour Saturation",
        "Colour Crosstalk", "Contrast Curve",
        "Sector Squash", "Sector Saturation",
        "Sector Brightness", "Sector Skew",
        "Highlight Bleach",
        "Neutral Tint", "Neutral Tint",
        "Colour Saturation",
    ],
}


from app.core.chromogen import CHROMOGEN_STAGES  # noqa: E402

for _cls in CHROMOGEN_STAGES:
    STAGE_POOL[_cls.name] = _cls
