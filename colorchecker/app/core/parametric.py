"""Parametric match solver: an ordered chain of stages fitted to
patch pairs — stagewise initialization, then a joint bounded
least-squares refine over the concatenated parameter vector.

Shares the pair preparation (NaN dropping, DRT sandwich inversion)
with the RBF path via match.prepare_pairs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from app.core.lut import CubeLUT, apply_lut
from app.core.match import invert_lut_at
from app.core.stages import Stage


@dataclass
class ParametricModel:
    stages: list[Stage]
    params: list[np.ndarray]  # one vector per stage, chain order
    strength: float = 1.0

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(rgb, dtype=np.float64)
        out = x
        for stage, p in zip(self.stages, self.params):
            out = stage.apply(out, p)
        if self.strength != 1.0:
            out = x + self.strength * (out - x)
        return out


@dataclass
class ParametricResult:
    model: ParametricModel
    pairs_used: int
    pairs_dropped: int
    pairs_unreachable: int
    display_referred: bool
    error_before: float
    error_after: float
    error_after_max: float
    per_patch_error: np.ndarray
    waterfall: list  # [(stage name, error after that stage)]
    stage_reports: list  # human-readable per-stage summaries


def _mean_dist(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


def solve_parametric(
    source: np.ndarray,
    target: np.ndarray,
    stages: list[Stage],
    strength: float = 1.0,
    output_transform: CubeLUT | None = None,
    regularization: float = 1e-3,
    sweeps: int = 2,
) -> ParametricResult:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            "source and target must be (N, 3) arrays of equal length — "
            f"got {source.shape} vs {target.shape}"
        )
    if not stages:
        raise ValueError("The stage chain is empty — add at least one stage")

    valid = ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1))
    dropped = int((~valid).sum())
    source, target = source[valid], target[valid]

    unreachable = 0
    if output_transform is not None:
        fit_target, reachable, _ = invert_lut_at(output_transform, target)
        unreachable = int((~reachable).sum())
        source, target = source[reachable], target[reachable]
        fit_target = fit_target[reachable]
    else:
        fit_target = target

    if source.shape[0] < 8:
        raise ValueError("Need at least 8 valid patch pairs to fit")

    def display(values):
        return apply_lut(output_transform, values) if output_transform is not None else values

    params = [stage.identity().astype(np.float64) for stage in stages]

    def chain(x, plist):
        out = x
        for stage, p in zip(stages, plist):
            out = stage.apply(out, p)
        return out

    # ---- pass 1: stagewise coordinate descent -------------------------
    for _ in range(sweeps):
        for i, stage in enumerate(stages):
            lo, hi = stage.bounds()
            x0 = np.clip(params[i], lo, hi)

            def residual(p, i=i):
                trial = params[:i] + [p] + params[i + 1 :]
                return (chain(source, trial) - fit_target).ravel()

            sol = least_squares(
                residual, x0, bounds=(lo, hi), method="trf",
                xtol=1e-10, ftol=1e-10, max_nfev=200,
            )
            params[i] = sol.x

    # ---- pass 2: joint refine with identity regularization ------------
    sizes = [p.size for p in params]
    offsets = np.cumsum([0] + sizes)
    lo_all = np.concatenate([s.bounds()[0] for s in stages])
    hi_all = np.concatenate([s.bounds()[1] for s in stages])
    identity_all = np.concatenate([s.identity() for s in stages])
    scale_all = np.maximum(hi_all - lo_all, 1e-6)
    reg_weight = np.sqrt(regularization)

    def split(flat):
        return [flat[offsets[i] : offsets[i + 1]] for i in range(len(stages))]

    def joint_residual(flat):
        fit = chain(source, split(flat)) - fit_target
        reg = reg_weight * (flat - identity_all) / scale_all
        return np.concatenate([fit.ravel(), reg])

    x0 = np.clip(np.concatenate(params), lo_all, hi_all)
    sol = least_squares(
        joint_residual, x0, bounds=(lo_all, hi_all), method="trf",
        xtol=1e-10, ftol=1e-10, max_nfev=400,
    )
    params = [p.copy() for p in split(sol.x)]

    # ---- report -------------------------------------------------------
    model = ParametricModel(stages=stages, params=params, strength=strength)
    error_before = _mean_dist(display(source), target)

    waterfall = []
    out = source
    for stage, p in zip(stages, params):
        out = stage.apply(out, p)
        waterfall.append((stage.name, _mean_dist(display(out), target)))

    fitted = display(model(source))
    per_patch = np.linalg.norm(fitted - target, axis=1)

    return ParametricResult(
        model=model,
        pairs_used=int(source.shape[0]),
        pairs_dropped=dropped,
        pairs_unreachable=unreachable,
        display_referred=output_transform is not None,
        error_before=error_before,
        error_after=float(per_patch.mean()),
        error_after_max=float(per_patch.max()),
        per_patch_error=per_patch,
        waterfall=waterfall,
        stage_reports=[s.describe(p) for s, p in zip(stages, params)],
    )
