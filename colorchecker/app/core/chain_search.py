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

from app.core.chromogen import CHROMOGEN_STAGES
from app.core.lut import CubeLUT
from app.core.match import invert_lut_at
from app.core.parametric import ParametricResult, solve_parametric
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
                   regularization: float, max_nfev: int = 80):
    """Fit ONE stage on top of the frozen chain output `cur`. Returns
    (params, fit_error). Multi-starts the Hue slider when present."""
    lo, hi = stage.bounds()
    identity = stage.identity()
    scale = np.maximum(hi - lo, 1e-6)
    reg = np.sqrt(regularization * stage.reg_scale)

    def residual(p):
        fit = (stage.apply(cur, p) - fit_target).ravel()
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
        err = _fit_err(stage.apply(cur, sol.x), fit_target)
        if err < best_e:
            best_p, best_e = sol.x, err
    return best_p, best_e


def _joint_refine(stages, params, source, fit_target, regularization,
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
        fit = (out - fit_target).ravel()
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
    output_transform: CubeLUT | None = None,
    regularization: float = 1e-3,
    backend: str = "scipy",
    verbose: bool = False,
) -> ParametricResult:
    """Search a chain of at most `max_nodes` nodes drawn freely (with
    repetition) from `pool`, then polish + report via solve_parametric.
    `min_gain` is the relative fit-error improvement a new node must
    deliver to be accepted (0.005 = 0.5%)."""
    if pool is None:
        pool = default_pool()
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # pair prep, mirroring solve_parametric (which re-does it for the
    # final report on the same inputs — semantics stay identical)
    valid = ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1))
    src, tgt = source[valid], target[valid]
    if output_transform is not None:
        fit_target, reachable, _ = invert_lut_at(output_transform, tgt)
        src, fit_target = src[reachable], fit_target[reachable]
    else:
        fit_target = tgt

    stages: list[Stage] = []
    params: list[np.ndarray] = []
    cur = src
    err = _fit_err(cur, fit_target)
    log = []

    while len(stages) < max_nodes:
        best = None  # (err, stage, params)
        for cls in pool:
            stage = cls()
            p, e = _fit_candidate(stage, cur, fit_target, regularization)
            if best is None or e < best[0]:
                best = (e, stage, p)

        gain = (err - best[0]) / max(err, 1e-12)
        if gain < min_gain:
            log.append(f"stopped: best candidate ({best[1].name}) gains "
                       f"{gain * 100.0:.2f}% < {min_gain * 100.0:.2f}%")
            break

        stages.append(best[1])
        params.append(best[2])
        params = _joint_refine(stages, params, src, fit_target,
                               regularization)
        cur = src
        for stage, p in zip(stages, params):
            cur = stage.apply(cur, p)
        err = _fit_err(cur, fit_target)
        log.append((len(stages), best[1].name, err))
        if verbose:
            print(f"  node {len(stages)}: + {best[1].name}  "
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
        regularization=regularization,
        backend=backend,
        init_params=params,
    )
    result.search_log = log
    return result
