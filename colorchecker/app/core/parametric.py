"""Parametric match solver: an ordered chain of stages fitted to
patch pairs — stagewise initialization, then a joint bounded
least-squares refine over the concatenated parameter vector.

Shares the pair preparation (NaN dropping, DRT sandwich inversion)
with the RBF path via match.prepare_pairs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from app.core.diagnostics import noise_gain
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
    backend: str = "scipy"  # which optimizer produced the fit
    # artifact KPI: empirical noise amplification (see diagnostics.py),
    # per stage at its own input distribution + for the whole chain
    stage_noise_gain: list = None  # [(stage name, stats dict)]
    chain_noise_gain: dict = None
    stage_labels: list = None  # short human names ("cool lows", ...)
    # filled by chain_search: [(round, stage name, fit error after
    # accepting it)] plus the stop reason as a final string entry
    search_log: list = None


def _mean_dist(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


def validate_backend(backend: str) -> None:
    """Raise early if the requested backend cannot run — callers doing
    expensive work first (the chain search) must fail BEFORE it."""
    if backend not in ("scipy", "torch"):
        raise ValueError(f"Unknown backend {backend!r} — use 'scipy' or 'torch'")
    if backend == "torch":
        from app.core.backprop import torch_available

        if not torch_available():
            raise RuntimeError(
                "backend='torch' needs PyTorch — install it with "
                "python3 -m pip install torch (optional dependency), "
                "or use backend='scipy'"
            )


def solve_parametric(
    source: np.ndarray,
    target: np.ndarray,
    stages: list[Stage],
    strength: float = 1.0,
    output_transform: CubeLUT | None = None,
    regularization: float = 1e-3,
    sweeps: int = 2,
    backend: str = "scipy",
    init_params: list[np.ndarray] | None = None,
    display_transform=None,
    frozen: int = 0,
) -> ParametricResult:
    """backend='torch' inserts a gradient (backprop) refinement pass —
    Adam over autograd mirrors of the stages, with multi-restart hue
    placement for Reuleaux Fine zones — between the stagewise init and
    the scipy joint refine. Requires the optional PyTorch dependency.

    `init_params` warm-starts the solve (one vector per stage) instead
    of the identity start — used by the chain search to polish a chain
    it has already roughed in. The stagewise sweeps still run (each
    least_squares starts from the warm values, so they can only keep or
    improve the fit).

    `display_transform` (callable, e.g. app.core.opendrt.OpenDRTModel)
    is the ANALYTIC display-referred mode: `target` is display values
    and the residual is display_transform(chain(x)) - target, computed
    directly — no LUT inversion, so NO pairs are dropped as
    unreachable and clipped targets still pull the fit the right way.
    Mutually exclusive with `output_transform` (the cube-inversion
    sandwich). scipy backend only for now (no torch mirror yet).

    `frozen`: the first N stages keep their init_params EXACTLY — they
    are applied but never optimized (the chain search's grey-scale-
    locked tone node). Requires init_params."""
    validate_backend(backend)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            "source and target must be (N, 3) arrays of equal length — "
            f"got {source.shape} vs {target.shape}"
        )
    if not stages:
        raise ValueError("The stage chain is empty — add at least one stage")

    if output_transform is not None and display_transform is not None:
        raise ValueError("Pass output_transform (cube sandwich) OR "
                         "display_transform (analytic), not both")
    if display_transform is not None and backend == "torch":
        raise NotImplementedError(
            "display_transform has no torch mirror yet — use "
            "backend='scipy' with the analytic DRT"
        )

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
        # analytic display mode: fit straight against the display
        # targets through display_transform — nothing to invert, no
        # pairs dropped
        fit_target = target

    if source.shape[0] < 8:
        raise ValueError("Need at least 8 valid patch pairs to fit")

    fwd = display_transform if display_transform is not None else (lambda v: v)

    def display(values):
        if output_transform is not None:
            return apply_lut(output_transform, values)
        return fwd(values)

    if init_params is not None:
        if len(init_params) != len(stages):
            raise ValueError("init_params must have one vector per stage")
        params = [np.asarray(p, dtype=np.float64).copy() for p in init_params]
    else:
        # init() defaults to identity(); stages whose identity sits in a
        # dead-gradient region (the filmic curve's parked toe/shoulder)
        # start mid-engaged so the fit can discover those controls.
        params = [stage.init().astype(np.float64) for stage in stages]

    if frozen and init_params is None:
        raise ValueError("frozen stages need init_params")
    frozen = max(0, min(frozen, len(stages)))
    frozen_params = [p.copy() for p in params[:frozen]]
    opt_stages = stages[frozen:]
    opt_params = params[frozen:]
    src_opt = source
    for stage, p in zip(stages[:frozen], frozen_params):
        src_opt = stage.apply(src_opt, p)

    def chain(x, plist):
        out = x
        for stage, p in zip(opt_stages, plist):
            out = stage.apply(out, p)
        return out

    if opt_stages:
        # ---- pass 1: stagewise coordinate descent ---------------------
        # Strongly anchored prep stages (high reg_scale) are fitted LAST
        # in each sweep: the look stages get first claim on the residual,
        # so prep only absorbs what the look genuinely cannot express
        # (e.g. a real exposure/WB mismatch). Fitting chain-order instead
        # lets a front prep stage greedily grab the look and park the
        # solve in a bad valley. The identity reg breaks remaining ties.
        fit_order = sorted(range(len(opt_stages)),
                           key=lambda i: (opt_stages[i].reg_scale, i))
        for _ in range(sweeps):
            for i in fit_order:
                stage = opt_stages[i]
                lo, hi = stage.bounds()
                x0 = np.clip(opt_params[i], lo, hi)
                stage_id = stage.identity()
                stage_scale = np.maximum(hi - lo, 1e-6)
                stage_reg = np.sqrt(regularization * stage.reg_scale)

                def residual(p, i=i, stage_id=stage_id,
                             stage_scale=stage_scale, stage_reg=stage_reg):
                    trial = opt_params[:i] + [p] + opt_params[i + 1 :]
                    fit = (fwd(chain(src_opt, trial)) - fit_target).ravel()
                    reg = stage_reg * (p - stage_id) / stage_scale
                    return np.concatenate([fit, reg])

                sol = least_squares(
                    residual, x0, bounds=(lo, hi), method="trf",
                    xtol=1e-10, ftol=1e-10, max_nfev=200,
                )
                opt_params[i] = sol.x

        # ---- pass 1.5 (torch backend): backprop refine + placement ----
        if backend == "torch":
            from app.core.backprop import refine_backprop

            opt_params = refine_backprop(
                opt_stages, opt_params, src_opt, fit_target,
                regularization=regularization,
            )

        # ---- pass 2: joint refine with identity regularization --------
        sizes = [p.size for p in opt_params]
        offsets = np.cumsum([0] + sizes)
        lo_all = np.concatenate([s.bounds()[0] for s in opt_stages])
        hi_all = np.concatenate([s.bounds()[1] for s in opt_stages])
        identity_all = np.concatenate([s.identity() for s in opt_stages])
        scale_all = np.maximum(hi_all - lo_all, 1e-6)
        reg_weight = np.sqrt(regularization)
        # per-stage anchoring: prep stages (high reg_scale) only move
        # when it makes the fit a LOT easier
        reg_scale_all = np.concatenate([
            np.full(s.identity().size, np.sqrt(s.reg_scale))
            for s in opt_stages
        ])

        def split(flat):
            return [flat[offsets[i] : offsets[i + 1]]
                    for i in range(len(opt_stages))]

        def joint_residual(flat):
            fit = fwd(chain(src_opt, split(flat))) - fit_target
            reg = reg_weight * reg_scale_all * (flat - identity_all) / scale_all
            return np.concatenate([fit.ravel(), reg])

        x0 = np.clip(np.concatenate(opt_params), lo_all, hi_all)
        sol = least_squares(
            joint_residual, x0, bounds=(lo_all, hi_all), method="trf",
            xtol=1e-10, ftol=1e-10, max_nfev=400,
        )
        opt_params = [p.copy() for p in split(sol.x)]

    params = frozen_params + opt_params

    # ---- report -------------------------------------------------------
    model = ParametricModel(stages=stages, params=params, strength=strength)
    error_before = _mean_dist(display(source), target)

    waterfall = []
    stage_gains = []
    out = source
    for stage, p in zip(stages, params):
        stage_gains.append(
            (stage.name, noise_gain(lambda v: stage.apply(v, p), out))
        )
        out = stage.apply(out, p)
        waterfall.append((stage.name, _mean_dist(display(out), target)))

    fitted = display(model(source))
    per_patch = np.linalg.norm(fitted - target, axis=1)

    return ParametricResult(
        model=model,
        pairs_used=int(source.shape[0]),
        pairs_dropped=dropped,
        pairs_unreachable=unreachable,
        display_referred=(output_transform is not None
                          or display_transform is not None),
        error_before=error_before,
        error_after=float(per_patch.mean()),
        error_after_max=float(per_patch.max()),
        per_patch_error=per_patch,
        waterfall=waterfall,
        stage_reports=[s.describe(p) for s, p in zip(stages, params)],
        backend=backend,
        stage_noise_gain=stage_gains,
        chain_noise_gain=noise_gain(model, source),
        stage_labels=[s.label(p) for s, p in zip(stages, params)],
    )
