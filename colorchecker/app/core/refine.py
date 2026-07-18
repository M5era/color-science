"""Grid refinement: snap sample squares onto the actual patches.

After corner detection the outer quad is known, but the patch grid's
inset (margins) still has to match the physical chart. This module
rectifies the quad to an axis-aligned image and searches margin values
that minimize the color variance inside every sample square — squares
centered on uniform patches score best, squares straddling patch edges
score worst. Geometry only: sampled VALUES always come from the raw
buffer via the sampler, never from the rectified preview.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.homography import homography_from_corners

_CELL_PX = 24  # rectified pixels per grid cell
# Score a larger area than we sample so misalignment is penalized sharply:
# squares this size straddle patch edges as soon as margins drift.
_SCORE_PATCH_FRACTION = 0.72


@dataclass
class RefineResult:
    margin_x: float  # % — same convention as the sidebar fields
    margin_y: float
    score: float


def _rectify(pixels: np.ndarray, corners: list[list[float]], width: int, height: int) -> np.ndarray:
    """Warp the quad to an axis-aligned (height, width) grayscale image,
    percentile-stretched for scoring."""
    H = homography_from_corners(corners)
    scale = np.array(
        [[1.0 / width, 0, 0], [0, 1.0 / height, 0], [0, 0, 1]], dtype=np.float64
    )
    # dst pixel -> unit square -> image pixel; WARP_INVERSE_MAP wants dst->src.
    matrix = H @ scale
    gray = pixels.mean(axis=2).astype(np.float32)
    rect = cv2.warpPerspective(
        gray, matrix.astype(np.float64), (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    lo, hi = np.percentile(rect, (1.0, 99.0))
    if hi - lo < 1e-9:
        return np.zeros_like(rect)
    return np.clip((rect - lo) / (hi - lo), 0.0, 1.0)


def _grid_variance_score(
    integral: np.ndarray, integral_sq: np.ndarray,
    rows: int, cols: int, width: int, height: int,
    margin_x: float, margin_y: float,
) -> float:
    """Sum of per-sample-square variance for one margin candidate.
    Uses integral images, so each square is O(1)."""
    mx = margin_x * width
    my = margin_y * height
    cell_w = (width - 2 * mx) / cols
    cell_h = (height - 2 * my) / rows
    if cell_w < 4 or cell_h < 4:
        return np.inf
    half_w = cell_w * _SCORE_PATCH_FRACTION / 2
    half_h = cell_h * _SCORE_PATCH_FRACTION / 2

    cxs = mx + (np.arange(cols) + 0.5) * cell_w
    cys = my + (np.arange(rows) + 0.5) * cell_h
    x0 = np.clip((cxs - half_w).astype(int), 0, width - 1)
    x1 = np.clip((cxs + half_w).astype(int), 1, width)
    y0 = np.clip((cys - half_h).astype(int), 0, height - 1)
    y1 = np.clip((cys + half_h).astype(int), 1, height)

    n = np.outer(y1 - y0, x1 - x0).astype(np.float64)
    if (n <= 0).any():
        return np.inf
    s = (
        integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)]
        - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)]
    )
    sq = (
        integral_sq[np.ix_(y1, x1)] - integral_sq[np.ix_(y0, x1)]
        - integral_sq[np.ix_(y1, x0)] + integral_sq[np.ix_(y0, x0)]
    )
    variances = np.maximum(sq / n - (s / n) ** 2, 0.0)
    return float(variances.sum())


def refine_margins(
    pixels: np.ndarray,
    corners: list[list[float]],
    rows: int,
    cols: int,
    max_margin_pct: float = 20.0,
    step_pct: float = 0.25,
) -> RefineResult:
    """Find margin_x/margin_y (%) that align the grid with the patches."""
    width, height = cols * _CELL_PX, rows * _CELL_PX
    rect = _rectify(pixels, corners, width, height)

    rect64 = rect.astype(np.float64)
    integral = cv2.integral(rect64)
    integral_sq = cv2.integral(rect64 * rect64)

    def search(xs: np.ndarray, ys: np.ndarray, seed: "RefineResult") -> "RefineResult":
        best = seed
        for my in ys:
            for mx in xs:
                score = _grid_variance_score(
                    integral, integral_sq, rows, cols, width, height, mx, my
                )
                if score < best.score:
                    best = RefineResult(mx * 100.0, my * 100.0, score)
        return best

    # Coarse pass over the full range, fine pass around the winner.
    coarse = np.arange(0.0, max_margin_pct + 1e-9, 1.0) / 100.0
    best = search(coarse, coarse, RefineResult(0.0, 0.0, np.inf))
    fine_x = np.arange(
        max(best.margin_x - 1.0, 0.0), best.margin_x + 1.0 + 1e-9, step_pct
    ) / 100.0
    fine_y = np.arange(
        max(best.margin_y - 1.0, 0.0), best.margin_y + 1.0 + 1e-9, step_pct
    ) / 100.0
    return search(fine_x, fine_y, best)
