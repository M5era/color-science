"""Projective mapping from unit chart space to image-pixel quads.

Pure numpy — no OpenCV dependency for the forward mapping, so the
geometry is testable without native libraries.
"""

import numpy as np

_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def homography_from_corners(corners: list[list[float]]) -> np.ndarray:
    """3x3 homography H mapping (u, v, 1) in the unit square to image pixels.

    `corners` is [TL, TR, BR, BL] in image coordinates, matching the unit
    square corners (0,0), (1,0), (1,1), (0,1).
    """
    src = _UNIT_SQUARE
    dst = np.asarray(corners, dtype=np.float64)
    if dst.shape != (4, 2):
        raise ValueError(f"Expected 4 corner points, got {dst.shape}")

    # Standard DLT for 4 correspondences: solve A h = 0.
    rows = []
    for (u, v), (x, y) in zip(src, dst):
        rows.append([u, v, 1, 0, 0, 0, -u * x, -v * x, -x])
        rows.append([0, 0, 0, u, v, 1, -u * y, -v * y, -y])
    A = np.asarray(rows)
    _, _, vt = np.linalg.svd(A)
    h = vt[-1]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) < 1e-12 or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Degenerate corner configuration")
    return H / H[2, 2]


def map_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply H to an (N, 2) array of (u, v) points -> (N, 2) image pixels."""
    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    mapped = homogeneous @ H.T
    return mapped[:, :2] / mapped[:, 2:3]


def patch_quads_image(overlay) -> list[tuple[int, int, np.ndarray]]:
    """Every patch's sample quad in image pixels: (row, col, (4, 2) array)."""
    H = homography_from_corners(overlay.corners)
    result = []
    for row, col, quad in overlay.patch_quads_unit():
        result.append((row, col, map_points(H, np.asarray(quad))))
    return result
