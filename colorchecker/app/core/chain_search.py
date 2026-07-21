"""Free-order chain search: FIND the chain instead of prescribing it.

Marc's 2026-07-21 rework of the matching pipeline: no Lift Gamma Gain,
no preset node order, every tool type usable as often as it helps —
the solver's only structural constraint is a maximum node count.

Greedy forward construction:

  1. Start from the empty chain.
  2. Each round, audition every stage type in the pool as the NEXT
     node: fit only the candidate's own parameters against the current
     residual (the existing chain's output is a fixed input, so each
     audition is a cheap small least-squares). Stages with a Hue
     slider are auditioned from several hue seeds — a mis-seeded hue
     window sees zero gradient and would otherwise never move.
  3. Append the winner, then jointly re-refine the WHOLE chain (all
     parameters, bounded, identity-regularized) so earlier nodes can
     hand work over to the newcomer.
  4. Stop at max_nodes, or when the best audition improves the fit
     error by less than min_gain (relative) — extra nodes that buy
     nothing stay out.
  5. Polish + report through solve_parametric (warm-started with the
     found parameters; backend='torch' adds the backprop pass there).

The order of the chain is therefore discovered, not prescribed: each
node was chosen because it reduced the residual best at its position.
"""

import numpy as np
from scipy.optimize import least_squares

from app.core.chromogen import CHROMOGEN_STAGES, ContrastCurveStage
from app.core.lut import CubeLUT
from app.core.match import invert_lut_at
from app.core.parametric import (
    ParametricResult,
    solve_parametric,
    validate_backend,
)
from app.core.stages import Stage

# hue seeds for auditioning stages that have a "Hue" slider (degrees)
_HUE_SEEDS = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)


def default_pool() -> list[type]:
    """The searchable node types: all Chromogen-family tools, NO Lift
    Gamma Gain (Marc's directive)."""
    return list(CHROMOGEN_STAGES)


def _fit_err(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


def _fit_candidate(stage: Stage, cur: np.ndarray, fit_target: np.ndarray,
                   regularization: float, fwd, max_nfev: int = 80):
    """Fit ONE stage on top of the frozen chain output `cur`. Returns
    (params, fit_error). Multi-starts the Hue slider when present.
    `fwd` maps working-domain output to the fit domain (identity, or
    the analytic display transform)."""
    lo, hi = stage.bounds()
    identity = stage.identity()
    scale = np.maximum(hi - lo, 1e-6)
    reg = np.sqrt(regularization * stage.reg_scale)

    def residual(p):
        fit = (fwd(stage.apply(cur, p)) - fit_target).ravel()
        return np.concatenate([fit, reg * (p - identity) / scale])

    starts = [identity]
    if "Hue" in stage.param_names:
        hue_i = stage.param_names.index("Hue")
        starts = []
        for seed in _HUE_SEEDS:
            s = identity.copy()
            s[hue_i] = seed
            starts.append(s)

    best_p, best_e = None, np.inf
    for x0 in starts:
        sol = least_squares(
            residual, np.clip(x0, lo, hi), bounds=(lo, hi), method="trf",
            xtol=1e-9, ftol=1e-9, max_nfev=max_nfev,
        )
        err = _fit_err(fwd(stage.apply(cur, sol.x)), fit_target)
        if err < best_e:
            best_p, best_e = sol.x, err
    return best_p, best_e


def _joint_refine(stages, params, source, fit_target, regularization, fwd,
                  max_nfev: int = 150):
    """Bounded joint least-squares over every parameter of the chain
    (same identity regularization as solve_parametric's pass 2)."""
    sizes = [p.size for p in params]
    offsets = np.cumsum([0] + sizes)
    lo = np.concatenate([s.bounds()[0] for s in stages])
    hi = np.concatenate([s.bounds()[1] for s in stages])
    identity = np.concatenate([s.identity() for s in stages])
    scale = np.maximum(hi - lo, 1e-6)
    reg = np.sqrt(regularization) * np.concatenate([
        np.full(s.identity().size, np.sqrt(s.reg_scale)) for s in stages
    ])

    def split(flat):
        return [flat[offsets[i]: offsets[i + 1]] for i in range(len(stages))]

    def residual(flat):
        out = source
        for stage, p in zip(stages, split(flat)):
            out = stage.apply(out, p)
        fit = (fwd(out) - fit_target).ravel()
        return np.concatenate([fit, reg * (flat - identity) / scale])

    sol = least_squares(
        residual, np.clip(np.concatenate(params), lo, hi),
        bounds=(lo, hi), method="trf",
        xtol=1e-9, ftol=1e-9, max_nfev=max_nfev,
    )
    return split(sol.x)


def search_chain(
    source: np.ndarray,
    target: np.ndarray,
    pool: list[type] | None = None,
    max_nodes: int = 10,
    min_gain: float = 0.005,
    broad_bias: float = 0.15,
    neutral_tone: bool = True,
    output_transform: CubeLUT | None = None,
    display_transform=None,
    regularization: float = 1e-3,
    backend: str = "scipy",
    verbose: bool = False,
) -> ParametricResult:
    """Search a chain of at most `max_nodes` nodes drawn freely (with
    repetition) from `pool`, then polish + report via solve_parametric.
    `min_gain` is the relative fit-error improvement a new node must
    deliver to be accepted (0.005 = 0.5%).

    `broad_bias` gives the BROAD tools a slight preference (Marc,
    2026-07-21: "so much sector stuff"): auditions by single-hue tools
    (stage.local_tool — the Sector family, Fine zones) have their gain
    discounted by this fraction when picking the round's winner, so a
    sector move must beat the best broad move by that margin to be
    chosen. 0 disables; the ACCEPTANCE test (min_gain) always uses the
    winner's real, undiscounted gain.

    `neutral_tone` (Marc, 2026-07-21: "contrast adjusted based on grey
    scale only"): before the free search, ONE Contrast Curve is fitted
    against the NEUTRAL samples only and FROZEN as node 1 — the grey
    scale sets the tone, the search explains color on top, and since
    every other pool tool is neutral-safe by construction the grey
    match can never be disturbed. Contrast Curve leaves the audition
    pool in this mode.

    `display_transform` (callable) switches to the ANALYTIC
    display-referred mode: `target` is display values, every audition
    and refine minimizes display_transform(chain(x)) - target directly
    — no cube inversion, no unreachable-dropping (the fix for tone
    evidence getting deleted at the extremes). scipy backend only."""
    # fail BEFORE the expensive search, not at the final polish (a
    # 20-node run once died at the very end on a missing torch)
    validate_backend(backend)
    if display_transform is not None and backend == "torch":
        raise NotImplementedError(
            "display_transform has no torch mirror yet — use "
            "backend='scipy' with the analytic DRT"
        )
    if pool is None:
        pool = default_pool()
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # pair prep, mirroring solve_parametric (which re-does it for the
    # final report on the same inputs — semantics stay identical)
    valid = ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1))
    src, tgt = source[valid], target[valid]
    if output_transform is not None and display_transform is not None:
        raise ValueError("Pass output_transform OR display_transform, "
                         "not both")
    if output_transform is not None:
        fit_target, reachable, _ = invert_lut_at(output_transform, tgt)
        src, fit_target = src[reachable], fit_target[reachable]
    else:
        fit_target = tgt

    fwd = display_transform if display_transform is not None else (lambda v: v)

    stages: list[Stage] = []
    params: list[np.ndarray] = []
    frozen_n = 0
    log = []
    cur = src

    # ---- grey-scale-locked tone: fit ONE Contrast Curve on the
    # neutral samples only and freeze it as node 1
    if neutral_tone:
        neutral = np.all(src == src[:, :1], axis=1)
        if neutral.sum() >= 8:
            con = ContrastCurveStage()
            p, _ = _fit_candidate(con, src[neutral], fit_target[neutral],
                                  regularization, fwd, max_nfev=200)
            stages.append(con)
            params.append(p)
            frozen_n = 1
            pool = [cls for cls in pool
                    if not issubclass(cls, ContrastCurveStage)]
            cur = con.apply(src, p)
            err0 = _fit_err(fwd(cur), fit_target)
            log.append((1, "Contrast Curve [grey-scale-locked tone]", err0))
            if verbose:
                print(f"  node 1: + Contrast Curve (grey-scale-locked "
                      f"tone)  fit error -> {err0:.5f}")
        else:
            log.append("neutral_tone skipped: no neutral samples in source")

    src_after_frozen = cur
    err = _fit_err(fwd(cur), fit_target)

    while len(stages) < max_nodes:
        best = None       # winner by DISCOUNTED gain
        best_real = None  # winner by real gain (fallback for the stop test)
        for cls in pool:
            stage = cls()
            p, e = _fit_candidate(stage, cur, fit_target, regularization, fwd)
            gain = err - e
            adj = gain * (1.0 - broad_bias) if stage.local_tool else gain
            if best is None or adj > best[0]:
                best = (adj, e, stage, p)
            if best_real is None or gain > best_real[0]:
                best_real = (gain, e, stage, p)

        if (err - best[1]) / max(err, 1e-12) < min_gain:
            # the biased winner stalls — fall back to the raw best so
            # the bias can never stop a search that still has real gains
            best = (None, *best_real[1:])
        gain = (err - best[1]) / max(err, 1e-12)
        if gain < min_gain:
            log.append(f"stopped: best candidate ({best[2].name}) gains "
                       f"{gain * 100.0:.2f}% < {min_gain * 100.0:.2f}%")
            break

        stages.append(best[2])
        params.append(best[3])
        # refine everything EXCEPT the frozen grey-locked tone node
        refined = _joint_refine(stages[frozen_n:], params[frozen_n:],
                                src_after_frozen, fit_target,
                                regularization, fwd)
        params = params[:frozen_n] + refined
        cur = src
        for stage, p in zip(stages, params):
            cur = stage.apply(cur, p)
        err = _fit_err(fwd(cur), fit_target)
        log.append((len(stages), best[2].name, err))
        if verbose:
            print(f"  node {len(stages)}: + {best[2].name}  "
                  f"fit error -> {err:.5f}")
    else:
        log.append(f"stopped: max_nodes ({max_nodes}) reached")

    if not stages:
        raise ValueError(
            "Chain search found nothing worth adding — the target is "
            "already within min_gain of the source (or min_gain is too "
            "strict)"
        )

    result = solve_parametric(
        source, target, stages,
        output_transform=output_transform,
        display_transform=display_transform,
        regularization=regularization,
        backend=backend,
        init_params=params,
        frozen=frozen_n,
    )
    result.search_log = log
    return result
