"""Gradient (backprop) refinement for the parametric solver.

Optional torch backend: autograd gradients through the stage mirrors
(app/core/torch_stages.py) instead of scipy's finite differences.

What it buys beyond the scipy path:

- ZONE PLACEMENT. A Reuleaux Fine zone whose window does not overlap
  the residual sees zero finite-difference gradient — scipy leaves it
  parked at its start hue. Here each restart re-seeds every Fine
  stage's hue center at a different spot on the wheel and the best
  final loss wins, so zones can be *found*, not just polished.
- Exact gradients through the C1-smooth windows (they were built for
  this — no kinks anywhere in the fine stage).

Box bounds are enforced by construction: parameters live as
unconstrained tensors mapped through a scaled sigmoid into [lo, hi],
so the optimizer never needs projection or clipping.

The caller (parametric.solve_parametric) runs its scipy joint refine
AFTER this, so the torch backend can only improve on scipy: placement
comes from here, final polish from the same least-squares as always.
"""

import numpy as np

from app.core.stages import ReuleauxFineStage, Stage


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def refine_backprop(
    stages: list[Stage],
    params: list[np.ndarray],
    source: np.ndarray,
    fit_target: np.ndarray,
    regularization: float = 1e-3,
    restarts: int = 6,
    iterations: int = 400,
    lr: float = 0.03,
    seed: int = 7,
) -> list[np.ndarray]:
    """Refine `params` (one vector per stage) by Adam on the full chain.

    Restarts only vary the hue centers of Reuleaux Fine stages (all
    other parameters restart from the given init); if the chain has no
    Fine stage a single run is done — restarts would all be identical.
    Returns the parameter vectors of the best run (data loss only).
    """
    import torch

    from app.core.torch_stages import torch_chain

    lo = np.concatenate([s.bounds()[0] for s in stages])
    hi = np.concatenate([s.bounds()[1] for s in stages])
    identity = np.concatenate([s.identity() for s in stages])
    scale = np.maximum(hi - lo, 1e-6)
    sizes = [p.size for p in params]
    offsets = np.cumsum([0] + sizes)

    fine_hue_indices = [
        offsets[i]
        for i, stage in enumerate(stages)
        if isinstance(stage, ReuleauxFineStage)
    ]
    if not fine_hue_indices:
        restarts = 1

    reg_scale = np.concatenate([
        np.full(s.identity().size, s.reg_scale) for s in stages
    ])

    lo_t = torch.as_tensor(lo, dtype=torch.float64)
    scale_t = torch.as_tensor(scale, dtype=torch.float64)
    id_t = torch.as_tensor(identity, dtype=torch.float64)
    reg_scale_t = torch.as_tensor(reg_scale, dtype=torch.float64)
    x_t = torch.as_tensor(source, dtype=torch.float64)
    y_t = torch.as_tensor(fit_target, dtype=torch.float64)
    reg = float(regularization)

    def to_theta(p_flat):
        frac = np.clip((p_flat - lo) / scale, 1e-4, 1.0 - 1e-4)
        return np.log(frac / (1.0 - frac))

    def split(p):
        return [p[offsets[i] : offsets[i + 1]] for i in range(len(stages))]

    rng = np.random.default_rng(seed)
    init = np.concatenate(params)
    best_loss, best_params = np.inf, None

    for restart in range(restarts):
        start = init.copy()
        if restart > 0:
            # spread Fine zones around the wheel: even coverage + jitter
            for k, idx in enumerate(fine_hue_indices):
                start[idx] = (
                    (restart - 1 + k / max(len(fine_hue_indices), 1))
                    / max(restarts - 1, 1)
                    + rng.uniform(-0.08, 0.08)
                ) % 1.0

        theta = torch.tensor(to_theta(start), requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=lr)

        for _ in range(iterations):
            optimizer.zero_grad()
            p = lo_t + scale_t * torch.sigmoid(theta)
            out = torch_chain(stages, x_t, split(p))
            data = ((out - y_t) ** 2).mean()
            penalty = reg * (reg_scale_t * ((p - id_t) / scale_t) ** 2).mean()
            (data + penalty).backward()
            optimizer.step()

        with torch.no_grad():
            p = lo_t + scale_t * torch.sigmoid(theta)
            out = torch_chain(stages, x_t, split(p))
            loss = float(((out - y_t) ** 2).mean())
        if loss < best_loss:
            best_loss = loss
            best_params = p.detach().numpy().copy()

    return [v.copy() for v in split(best_params)]
