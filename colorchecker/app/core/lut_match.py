"""Match the parametric stage chain to a LUT (Plan C item 1).

A LUT is a function, so no footage is needed: sample source points in
the LUT's domain, apply the LUT to get targets, and those pairs feed
solve_parametric exactly like measured patches. Result: "this LUT
explained as Chromogen-style moves" — per-stage waterfall, noise-gain
KPI, paste-ready DCTL sliders.

Sampling: by default a jittered lattice across the LUT domain plus a
neutral-axis run (looks live or die on their grey behavior, so neutrals
are always represented). Alternatively pass explicit source points —
e.g. Marc's measured LogC3 patch dataset, which weights the fit toward
colors that actually occur on real charts.
"""

import numpy as np

from app.core.lut import CubeLUT, apply_lut
from app.core.parametric import ParametricResult, solve_parametric
from app.core.stages import Stage


def sample_lut_domain(lut: CubeLUT, n: int = 1500, seed: int = 11) -> np.ndarray:
    """Jittered-lattice sample of the LUT's domain + a neutral ramp."""
    rng = np.random.default_rng(seed)
    lo, hi = float(lut.domain_min[0]), float(lut.domain_max[0])

    per_axis = max(int(round(n ** (1.0 / 3.0))), 3)
    grid = np.linspace(lo, hi, per_axis)
    pts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    step = (hi - lo) / max(per_axis - 1, 1)
    pts = pts + rng.uniform(-0.35, 0.35, pts.shape) * step
    pts = np.clip(pts, lo, hi)

    neutral = np.linspace(lo, hi, 64)[:, None].repeat(3, axis=1)
    return np.concatenate([pts, neutral])


def solve_lut_match(
    lut: CubeLUT,
    stages: list[Stage],
    source_points: np.ndarray | None = None,
    n_samples: int = 1500,
    backend: str = "scipy",
    regularization: float = 1e-3,
    seed: int = 11,
    drt: CubeLUT | None = None,
    target_is_display: bool = False,
) -> ParametricResult:
    """With `drt`, the match runs as the display-referred sandwich:
    the chain approximates the LUT in the working (log) domain, but
    targets the DRT inverts back from display, unreachable/clipped
    patches are dropped, and errors are reported THROUGH the DRT —
    what the eye sees. Stack the fitted chain BEFORE the DRT node."""
    if source_points is None:
        source_points = sample_lut_domain(lut, n=n_samples, seed=seed)
    source_points = np.asarray(source_points, dtype=np.float64)
    targets = apply_lut(lut, source_points)
    if drt is not None:
        # target_is_display: the LUT already renders to display (e.g. a
        # print emulation) — rebuild it as [chain under the DRT], i.e.
        # solve DRT(chain(x)) ~= lut(x). Otherwise the LUT is a log-
        # domain look to be viewed through the DRT.
        display_targets = targets if target_is_display else apply_lut(drt, targets)
        return solve_parametric(
            source_points, display_targets, stages,
            backend=backend, regularization=regularization,
            output_transform=drt,
        )
    return solve_parametric(
        source_points, targets, stages,
        backend=backend, regularization=regularization,
    )
