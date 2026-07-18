"""Patch sampling: mean raw RGB inside each patch's image-space quad.

Reads ONLY the raw float buffer from image_io — never the display
preview. The mean is computed in float64 over the exact pixels whose
centers fall inside the perspective-mapped sample polygon (a pixel at
index (i, j) has its center at (j + 0.5, i + 0.5) in image coordinates).
"""

from dataclasses import dataclass

import numpy as np

from app.core.homography import patch_quads_image
from app.core.overlay import Overlay


@dataclass
class PatchSample:
    row: int  # 1-indexed, matching the table
    col: int
    rgb: tuple[float, float, float]
    pixel_count: int

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "rgb": [float(v) for v in self.rgb],
            "pixel_count": self.pixel_count,
        }


def sample_overlay(pixels: np.ndarray, overlay: Overlay) -> list[PatchSample]:
    """Sample every patch of `overlay` from the raw float buffer.

    Returns row-major PatchSamples (the export-order invariant). Patches
    whose polygon lies outside the image sample zero pixels and return
    NaN values rather than silently sampling wrong data.
    """
    height, width = pixels.shape[:2]
    samples: list[PatchSample] = []

    for row, col, quad in patch_quads_image(overlay):
        x0 = int(np.floor(quad[:, 0].min()))
        x1 = int(np.ceil(quad[:, 0].max())) + 1
        y0 = int(np.floor(quad[:, 1].min()))
        y1 = int(np.ceil(quad[:, 1].max())) + 1
        x0c, x1c = max(x0, 0), min(x1, width)
        y0c, y1c = max(y0, 0), min(y1, height)

        if x1c <= x0c or y1c <= y0c:
            samples.append(PatchSample(row, col, (np.nan, np.nan, np.nan), 0))
            continue

        mask = _pixel_centers_in_convex_quad(quad, x0c, x1c, y0c, y1c)
        count = int(mask.sum())
        if count == 0:
            samples.append(PatchSample(row, col, (np.nan, np.nan, np.nan), 0))
            continue

        region = pixels[y0c:y1c, x0c:x1c].astype(np.float64)
        masked = region[mask]
        mean = masked.mean(axis=0)
        samples.append(
            PatchSample(row, col, (float(mean[0]), float(mean[1]), float(mean[2])), count)
        )

    return samples


def _pixel_centers_in_convex_quad(
    quad: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> np.ndarray:
    """Boolean mask over [y0:y1, x0:x1] of pixels whose centers lie inside
    the convex quad (homographies of squares stay convex when valid)."""
    xs = np.arange(x0, x1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1, dtype=np.float64) + 0.5
    cx, cy = np.meshgrid(xs, ys)

    inside = np.ones(cx.shape, dtype=bool)
    signs = []
    for i in range(4):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % 4]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        signs.append(cross)
    # Points must be on the same side of all four edges (either winding).
    all_pos = np.ones(cx.shape, dtype=bool)
    all_neg = np.ones(cx.shape, dtype=bool)
    for cross in signs:
        all_pos &= cross >= 0
        all_neg &= cross <= 0
    inside = all_pos | all_neg
    return inside
