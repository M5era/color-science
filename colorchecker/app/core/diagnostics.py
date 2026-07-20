"""Artifact KPIs for stages and fitted chains.

NOISE GAIN — the headline metric (Marc's request after Sector
Saturation visibly amplified sensor noise): how much a small random
perturbation of the input (i.e. noise) is amplified by a transform,
measured empirically at real working points.

    gain = |f(x + eps*d) - f(x)| / eps      (d = random unit direction)

gain ~ 1.0  -> transparent, noise passes through unchanged
gain >~ 2   -> the transform doubles noise there — expect visible
               amplification in flat areas
gain -> big -> the power-law-near-zero failure mode

Reported per stage (at that stage's actual input distribution) and for
the whole chain, as median / p95 / max over points x directions. This
is a directional finite-difference estimate of the Jacobian norm, so
it also flags kinks and steep cliffs — any "local contrast explosion"
an artifact would ride on.
"""

import numpy as np


def noise_gain(fn, points, eps: float = 0.002, n_directions: int = 4,
               seed: int = 0) -> dict:
    """Empirical amplification of a small perturbation by `fn`,
    measured at `points` (N, 3). Returns median/p95/max over
    N * n_directions probes."""
    points = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    base = fn(points)
    gains = []
    for _ in range(n_directions):
        d = rng.normal(size=points.shape)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        moved = fn(points + eps * d)
        gains.append(np.linalg.norm(moved - base, axis=1) / eps)
    g = np.concatenate(gains)
    return {
        "median": float(np.median(g)),
        "p95": float(np.percentile(g, 95)),
        "max": float(g.max()),
    }


def format_gain(stats: dict) -> str:
    return (f"noise gain ×{stats['median']:.2f} "
            f"(p95 ×{stats['p95']:.2f}, max ×{stats['max']:.2f})")
