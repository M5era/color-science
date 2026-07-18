"""Chart auto-detection: loose user rectangle -> precise chart quad.

The user drags a rough box around the chart; we look for the strongest
convex quadrilateral inside it. Log/flat footage has weak contrast, so
the crop is percentile-stretched before edge detection. If no plausible
quad is found the drawn rectangle itself is returned, so the tool
degrades to manual placement instead of failing.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectionResult:
    corners: list[list[float]]  # TL, TR, BR, BL in full-image pixels
    detected: bool  # False -> fell back to the drawn rectangle


def _order_corners(quad: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    ordered = quad[np.argsort(angles)]
    # argsort by angle yields counter-clockwise starting anywhere; rotate
    # so the point with the smallest x+y (top-left) comes first, then flip
    # to clockwise TL, TR, BR, BL.
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1][1] > ordered[-1][1]:  # second point should be the top edge
        ordered = np.roll(ordered[::-1], 1, axis=0)
    return ordered


def _stretch_to_u8(crop: np.ndarray) -> np.ndarray:
    """Percentile-normalize a raw float crop to uint8 for edge detection.

    Display-only path: detection output is geometry, never pixel values.
    """
    gray = crop.mean(axis=2)
    lo, hi = np.percentile(gray, (1.0, 99.0))
    if hi - lo < 1e-9:
        return np.zeros(gray.shape, dtype=np.uint8)
    stretched = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255.0 + 0.5).astype(np.uint8)


def _patch_field_quad(u8: np.ndarray, crop_area: float) -> np.ndarray | None:
    """Find the patch mosaic: many small square-ish blobs -> bounding quad.

    Far more robust on real log footage than chart-outline contours,
    because the chart border often blends into a dark background while
    the patch grid is always high-frequency structure.
    """
    block = max(int(np.sqrt(crop_area) / 8) | 1, 15)
    candidates = []
    for c_offset in (-4, 4):
        thresh_type = cv2.THRESH_BINARY if c_offset < 0 else cv2.THRESH_BINARY_INV
        th = cv2.adaptiveThreshold(
            u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, block, c_offset
        )
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (2e-4 * crop_area < area < 2e-2 * crop_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = bw / bh if bh else 0.0
            if not (0.4 < aspect < 2.5):
                continue
            if area / (bw * bh) < 0.5:  # not filled -> not a patch
                continue
            candidates.append(contour)

    if len(candidates) < 25:  # not enough patch-like blobs to trust
        return None

    # Patches share a size: drop blobs far from the median area (sticky
    # notes, light panels, sockets are much bigger or smaller).
    areas = np.array([cv2.contourArea(c) for c in candidates])
    median_area = float(np.median(areas))
    keep = (areas > 0.3 * median_area) & (areas < 3.0 * median_area)
    candidates = [c for c, k in zip(candidates, keep) if k]
    if len(candidates) < 25:
        return None

    # Patches form one dense grid: cluster blob centers by proximity and
    # keep the largest cluster, discarding spatially isolated clutter.
    centers = np.array(
        [c.reshape(-1, 2).mean(axis=0) for c in candidates], dtype=np.float64
    )
    dists = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    link = 2.5 * float(np.median(nn))
    n = len(centers)
    labels = np.full(n, -1)
    cluster = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = cluster
        while stack:
            j = stack.pop()
            for k in np.nonzero(dists[j] <= link)[0]:
                if labels[k] < 0:
                    labels[k] = cluster
                    stack.append(k)
        cluster += 1
    counts = np.bincount(labels)
    main = int(counts.argmax())
    if counts[main] < 25:
        return None
    candidates = [c for c, lbl in zip(candidates, labels) if lbl == main]

    points = np.vstack([c.reshape(-1, 2) for c in candidates])
    box = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
    return box.astype(np.float64)


def _border_quad(u8: np.ndarray, crop_area: float) -> np.ndarray | None:
    """Legacy path: strongest convex 4-corner contour (chart outline)."""
    blurred = cv2.GaussianBlur(u8, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_quad = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.2 * crop_area or area <= best_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best_quad = approx.reshape(4, 2).astype(np.float64)
            best_area = area
    return best_quad


def detect_chart_quad(
    pixels: np.ndarray,
    rect: tuple[float, float, float, float],
) -> DetectionResult:
    """Find the chart quad inside `rect` (x0, y0, x1, y1, image pixels).

    Tries the patch-mosaic detector first (robust on flat/log footage),
    then the chart-outline detector, then falls back to the drawn rect."""
    height, width = pixels.shape[:2]
    x0 = int(np.clip(min(rect[0], rect[2]), 0, width - 1))
    x1 = int(np.clip(max(rect[0], rect[2]), 0, width))
    y0 = int(np.clip(min(rect[1], rect[3]), 0, height - 1))
    y1 = int(np.clip(max(rect[1], rect[3]), 0, height))

    fallback = DetectionResult(
        corners=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], detected=False
    )
    if x1 - x0 < 8 or y1 - y0 < 8:
        return fallback

    crop = pixels[y0:y1, x0:x1]
    u8 = _stretch_to_u8(crop)
    crop_area = float((x1 - x0) * (y1 - y0))

    quad = _patch_field_quad(u8, crop_area)
    if quad is None:
        quad = _border_quad(u8, crop_area)
    if quad is None:
        return fallback

    ordered = _order_corners(quad)
    ordered += np.array([x0, y0], dtype=np.float64)
    return DetectionResult(corners=ordered.tolist(), detected=True)
