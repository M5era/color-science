"""Smooth mask windows for zone-based stages.

Shape: a plateau with raised-cosine (cos^2) shoulders — full weight
inside `flat` of the center, smooth falloff to zero over `soft`
beyond it. C1-continuous everywhere by construction, so masks cannot
introduce banding and gradient-based solvers see no kinks.

`plateau_window` works on a linear axis (luma, sat); `wrapped_window`
treats x as periodic with period 1 (hue in turns), so a window
centered near 0 correctly reaches across the 0/1 seam.

A window whose flat core covers the whole working domain has weight
1.0 everywhere there — that is the "mask off" state, and it is exactly
representable (used as the identity anchor by ReuleauxFineStage).
"""

import numpy as np

_MIN_SOFT = 1e-6


def plateau_window(x: np.ndarray, center: float, flat: float,
                   soft: float) -> np.ndarray:
    """Weight in [0, 1]: 1 for |x-center| <= flat, cos^2 falloff to 0
    at |x-center| >= flat + soft."""
    d = np.abs(np.asarray(x, dtype=np.float64) - center)
    return _shoulder(d, flat, soft)


def wrapped_window(x: np.ndarray, center: float, flat: float,
                   soft: float) -> np.ndarray:
    """plateau_window on a periodic axis of period 1 (hue in turns).
    Distance is the shorter way around the circle."""
    d = np.abs(((np.asarray(x, dtype=np.float64) - center + 0.5) % 1.0) - 0.5)
    return _shoulder(d, flat, soft)


def _shoulder(d: np.ndarray, flat: float, soft: float) -> np.ndarray:
    t = np.clip((d - flat) / max(soft, _MIN_SOFT), 0.0, 1.0)
    w = np.cos(0.5 * np.pi * t) ** 2
    # exact endpoints: cos(pi/2)^2 is ~4e-33, not 0 — pin the support
    # edges so "outside the window" means exactly untouched weight
    return np.where(t >= 1.0, 0.0, np.where(t <= 0.0, 1.0, w))
