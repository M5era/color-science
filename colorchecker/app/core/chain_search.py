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

from app.core.chromogen import (
    CHROMOGEN_STAGES,
    ContrastCurveStage,
    NeutralTintStage,
    SplitToneStage,
)
from app.core.filmic import FilmicContrastStage
from app.core.diagnostics import noise_gain
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

# how hard the un-frozen tone node is anchored to its grey-scale fit when
# a tint co-adapts with it (high = "move only as much as the tint needs").
# Strong: the tone should nudge for the tint's crossover, NOT wander off
# to a contorted exposure/mid-push combo chasing colour error (genesis
# drifted Exposure -0.66 -> -2.0 at reg 40).
_TONE_ANCHOR_REG = 250.0


def default_pool() -> list[type]:
    """The searchable node types: the Chromogen-family tools, NO Lift
    Gamma Gain and NO Contrast Curve — Filmic Contrast replaces it as
    the tone tool (Marc, 2026-07-22: "for the contrast, lets just use
    this for now, this really works"; it stays in STAGE_POOL for
    presets / manual use). Neutral Tint was out for a day in favour of
    Split Tone, then re-admitted alongside it once its falloff/pivot
    floors were fixed (Marc, same evening: "add it back in the ML")."""
    pool = [cls for cls in CHROMOGEN_STAGES
            if cls is not ContrastCurveStage]
    return pool + [FilmicContrastStage]


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

    # start from init(), not identity: stages whose identity sits in a
    # dead-gradient region (Filmic Contrast's parked white/black point)
    # seed engaged there; for every other stage init() IS identity
    starts = [stage.init()]
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
                  max_nfev: int = 150, anchors=None, reg_scales=None):
    """Bounded joint least-squares over every parameter of the chain
    (same identity regularization as solve_parametric's pass 2).

    `anchors` (list of per-stage arrays) overrides what each stage is
    regularized TOWARD — default is each stage's identity, but the tone
    node is anchored to its grey-scale fit when it co-adapts with a tint.
    `reg_scales` (list) overrides each stage's reg_scale (used to soft-
    anchor the un-frozen tone with a high weight)."""
    sizes = [p.size for p in params]
    offsets = np.cumsum([0] + sizes)
    lo = np.concatenate([s.bounds()[0] for s in stages])
    hi = np.concatenate([s.bounds()[1] for s in stages])
    if anchors is None:
        anchor = np.concatenate([s.identity() for s in stages])
    else:
        anchor = np.concatenate([np.asarray(a, np.float64) for a in anchors])
    if reg_scales is None:
        reg_scales = [s.reg_scale for s in stages]
    scale = np.maximum(hi - lo, 1e-6)
    reg = np.sqrt(regularization) * np.concatenate([
        np.full(s.identity().size, np.sqrt(rs))
        for s, rs in zip(stages, reg_scales)
    ])

    def split(flat):
        return [flat[offsets[i]: offsets[i + 1]] for i in range(len(stages))]

    def residual(flat):
        out = source
        for stage, p in zip(stages, split(flat)):
            out = stage.apply(out, p)
        fit = (fwd(out) - fit_target).ravel()
        return np.concatenate([fit, reg * (flat - anchor) / scale])

    sol = least_squares(
        residual, np.clip(np.concatenate(params), lo, hi),
        bounds=(lo, hi), method="trf",
        xtol=1e-9, ftol=1e-9, max_nfev=max_nfev,
    )
    return split(sol.x)


def _apply_chain(stages, params, x):
    out = x
    for stage, p in zip(stages, params):
        out = stage.apply(out, p)
    return out


def _refine_chain(stages, params, src, fit_target, regularization, fwd,
                  frozen_n, grey_anchor, unfreeze_on_tint, max_nfev=150):
    """Joint-refine a chain. Normally the frozen tone prefix is held and
    only stages[frozen_n:] move. But once a tint that is NOT neutral-safe
    (Split Tone or Neutral Tint — both can tint greys) is present and
    `unfreeze_on_tint` is set, the tone node is un-frozen INTO the joint
    solve, soft-anchored to its grey-scale fit, so tone and tint co-adapt
    on the (tinted) neutrals instead of the tone being locked-then-
    disturbed."""
    tint = any(isinstance(s, (NeutralTintStage, SplitToneStage))
               for s in stages[frozen_n:])
    # grey_anchor is None for a MANUAL prefix — those nodes are Marc's
    # hand-dialled values and must never be co-adapted
    if frozen_n and unfreeze_on_tint and tint and grey_anchor is not None:
        anchors = [grey_anchor] + [s.identity() for s in stages[frozen_n:]]
        reg_scales = ([_TONE_ANCHOR_REG]
                      + [s.reg_scale for s in stages[frozen_n:]])
        return _joint_refine(stages, params, src, fit_target,
                             regularization, fwd, max_nfev=max_nfev,
                             anchors=anchors, reg_scales=reg_scales)
    prefix = _apply_chain(stages[:frozen_n], params[:frozen_n], src)
    refined = _joint_refine(stages[frozen_n:], params[frozen_n:],
                            prefix, fit_target, regularization, fwd,
                            max_nfev=max_nfev)
    return list(params[:frozen_n]) + refined


def _prune_chain(stages, params, src, fit_target, regularization, fwd,
                 frozen_n, grey_anchor, unfreeze_on_tint, prune_tol, log,
                 prune_screen_k=4, verbose=False):
    """Drop redundant nodes. Each round: CHEAP-screen every node by the
    error its removal causes with NO re-fit (one forward pass), then
    re-refine only the `prune_screen_k` most-redundant candidates and drop
    the one that both stays within `prune_tol` and most REDUCES the
    chain's max noise gain (Marc: a node that barely helps the fit but
    spikes noise is the prime candidate). `prune_screen_k <= 0` disables
    the screen — every candidate is fully re-refined every round (the
    thorough, slow mode). Never prunes the frozen tone."""
    probe = src[:400]  # subsample for the noise-gain estimate (speed)
    full = prune_screen_k <= 0
    refine_nfev = 150 if full else 60   # thorough vs fast per-candidate refit
    rounds = 0
    while len(stages) > frozen_n + 1 and rounds < len(stages):
        rounds += 1
        base_err = _fit_err(fwd(_apply_chain(stages, params, src)), fit_target)
        # cheap screen: no-refit error of removing each node
        screen = []
        for i in range(frozen_n, len(stages)):
            ns = stages[:i] + stages[i + 1:]
            npar = params[:i] + params[i + 1:]
            e = _fit_err(fwd(_apply_chain(ns, npar, src)), fit_target)
            screen.append((e, i))
        screen.sort()
        candidates = screen if full else screen[:prune_screen_k]

        best = None  # (noise_max, err, idx, stages', params')
        for _, i in candidates:
            ns = stages[:i] + stages[i + 1:]
            npar = params[:i] + params[i + 1:]
            npar = _refine_chain(ns, npar, src, fit_target, regularization,
                                 fwd, frozen_n, grey_anchor,
                                 unfreeze_on_tint, max_nfev=refine_nfev)
            err = _fit_err(fwd(_apply_chain(ns, npar, src)), fit_target)
            if err <= base_err * (1.0 + prune_tol):
                ng = noise_gain(
                    lambda x: fwd(_apply_chain(ns, npar, x)), probe)["max"]
                if best is None or ng < best[0]:
                    best = (ng, err, i, ns, npar)
        if best is None:
            break
        _, err, i, ns, npar = best
        msg = (f"pruned node {i + 1} ({stages[i].name}) — "
               f"fit {base_err:.5f} -> {err:.5f}, noise max -> {best[0]:.1f}")
        log.append(msg)
        if verbose:
            print(f"  {msg}  ({len(ns)} nodes left)")
        stages, params = ns, npar
    return stages, params


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
    local_search: bool = False,
    prune_tol: float = 0.01,
    prune_screen_k: int = 4,
    pre_chain=None,
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
    evidence getting deleted at the extremes). scipy backend only.

    `local_search` turns the greedy builder into a light local search:
    (a) the frozen tone node is UN-FROZEN into the joint solve (soft-
    anchored to its grey fit) once a Neutral Tint is in the chain, so
    tone + tint co-adapt on tinted neutrals; and (b) after building, a
    PRUNE pass drops any node whose removal keeps the fit within
    `prune_tol` (default 1%), preferring the drop that most reduces the
    chain's max noise gain. Off by default so it can be A/B'd against
    the pure greedy path. `prune_screen_k` bounds the prune cost: each
    round only the K cheapest-to-drop nodes get the expensive re-refit
    (default 4); 0 = thorough mode, re-refit EVERY node every round."""
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
    grey_anchor = None   # the tone node's grey-scale fit (soft-anchor target)
    log = []
    cur = src

    # baseline: the error with NO nodes at all (source vs target)
    err0 = _fit_err(fwd(src), fit_target)
    log.append((0, "(no nodes / source)", err0))
    if verbose:
        print(f"  before any nodes: fit error -> {err0:.5f}")

    # ---- MANUAL PREFIX (Marc's 2026-07-22 workflow: "i would do the
    # contrast, exposure, and split myself first"): his hand-dialled
    # tone nodes are applied as a HARD-frozen prefix — never refit,
    # never unfrozen, never pruned — and the tone tools leave the
    # audition pool so the search only explains color on top.
    if pre_chain is not None:
        from app.core.filmic import ExposureStage
        pre_stages, pre_params = pre_chain
        stages = list(pre_stages)
        params = [np.asarray(p, dtype=np.float64).copy() for p in pre_params]
        frozen_n = len(stages)
        neutral_tone = False
        tone_types = (FilmicContrastStage, ContrastCurveStage,
                      SplitToneStage, ExposureStage)
        pool = [cls for cls in pool if not issubclass(cls, tone_types)]
        cur = _apply_chain(stages, params, src)
        err0 = _fit_err(fwd(cur), fit_target)
        log.append((frozen_n, f"[manual prefix: "
                    f"{' -> '.join(s.name for s in stages)}]", err0))
        if verbose:
            print(f"  manual prefix ({frozen_n} nodes: "
                  f"{' -> '.join(s.name for s in stages)})  "
                  f"fit error -> {err0:.5f}")

    # ---- grey-scale-locked tone: fit ONE tone node (Filmic Contrast)
    # on the neutral samples only and freeze it as node 1
    if neutral_tone:
        neutral = np.all(src == src[:, :1], axis=1)
        if neutral.sum() >= 8:
            con = FilmicContrastStage()
            p, _ = _fit_candidate(con, src[neutral], fit_target[neutral],
                                  regularization, fwd, max_nfev=200)
            stages.append(con)
            params.append(p)
            frozen_n = 1
            grey_anchor = p.copy()
            pool = [cls for cls in pool
                    if not issubclass(cls, (FilmicContrastStage,
                                            ContrastCurveStage))]
            cur = con.apply(src, p)
            err0 = _fit_err(fwd(cur), fit_target)
            log.append((1, "Filmic Contrast [grey-scale-locked tone]", err0))
            if verbose:
                print(f"  node 1: + Filmic Contrast (grey-scale-locked "
                      f"tone)  fit error -> {err0:.5f}")
        else:
            log.append("neutral_tone skipped: no neutral samples in source")

    err = _fit_err(fwd(cur), fit_target)

    # with a manual prefix, max_nodes counts the ADDED nodes (the prefix
    # is Marc's, not the search's)
    node_cap = max_nodes + (frozen_n if pre_chain is not None else 0)
    while len(stages) < node_cap:
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
        # refine the chain: the frozen tone is held, UNLESS local_search
        # un-freezes it to co-adapt with a tint (see _refine_chain)
        params = _refine_chain(stages, params, src, fit_target,
                               regularization, fwd, frozen_n, grey_anchor,
                               local_search)
        cur = _apply_chain(stages, params, src)
        err = _fit_err(fwd(cur), fit_target)
        log.append((len(stages), best[2].name, err))
        if verbose:
            print(f"  node {len(stages)}: + {best[2].name}  "
                  f"fit error -> {err:.5f}")
    else:
        log.append(f"stopped: max_nodes ({max_nodes}) reached")

    # local search: drop nodes that went redundant now the chain is built
    if local_search and len(stages) > frozen_n + 1:
        if verbose:
            mode = "full" if prune_screen_k <= 0 else f"screen {prune_screen_k}"
            print(f"  local search: pruning redundant nodes ({mode})...")
        stages, params = _prune_chain(
            stages, params, src, fit_target, regularization, fwd,
            frozen_n, grey_anchor, local_search, prune_tol, log,
            prune_screen_k=prune_screen_k, verbose=verbose)

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
