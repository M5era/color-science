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


def detect_chart_quad(
    pixels: np.ndarray,
    rect: tuple[float, float, float, float],
) -> DetectionResult:
    """Find the chart quad inside `rect` (x0, y0, x1, y1, image pixels)."""
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
    u8 = cv2.GaussianBlur(u8, (3, 3), 0)
    edges = cv2.Canny(u8, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = (x1 - x0) * (y1 - y0)
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

    if best_quad is None:
        return fallback

    ordered = _order_corners(best_quad)
    ordered += np.array([x0, y0], dtype=np.float64)
    return DetectionResult(corners=ordered.tolist(), detected=True)
