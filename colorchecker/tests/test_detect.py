"""Chart detection on synthetic flat-contrast frames."""

import numpy as np

from app.core.detect import detect_chart_quad


def _synthetic_chart_frame():
    """A dim, flat frame with a chart-like bordered patch grid, like log footage."""
    rng = np.random.default_rng(3)
    frame = np.full((300, 460, 3), 0.22, dtype=np.float32)
    frame += rng.normal(0, 0.003, frame.shape).astype(np.float32)
    # Chart: dark border frame with a grid of patches inside.
    x0, y0, x1, y1 = 90, 60, 330, 220
    frame[y0:y1, x0:x1] = 0.05
    inner = frame[y0 + 8 : y1 - 8, x0 + 8 : x1 - 8]
    rows, cols = 8, 12
    ph, pw = inner.shape[0] // rows, inner.shape[1] // cols
    for r in range(rows):
        for c in range(cols):
            value = rng.uniform(0.12, 0.55, size=3).astype(np.float32)
            inner[r * ph + 2 : (r + 1) * ph - 2, c * pw + 2 : (c + 1) * pw - 2] = value
    return frame, (x0, y0, x1, y1)


def test_detects_chart_inside_loose_rect():
    frame, (x0, y0, x1, y1) = _synthetic_chart_frame()
    # Loose selection: generous slop on every side.
    result = detect_chart_quad(frame, (x0 - 40, y0 - 30, x1 + 50, y1 + 35))
    assert result.detected
    corners = np.asarray(result.corners)
    expected = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    assert np.abs(corners - expected).max() < 8  # within a few pixels


def test_falls_back_to_rect_when_nothing_there():
    frame = np.full((100, 100, 3), 0.5, dtype=np.float32)
    result = detect_chart_quad(frame, (10, 20, 80, 90))
    assert not result.detected
    assert result.corners == [[10, 20], [80, 20], [80, 90], [10, 90]]


def test_corner_order_is_tl_tr_br_bl():
    frame, (x0, y0, x1, y1) = _synthetic_chart_frame()
    result = detect_chart_quad(frame, (x0 - 30, y0 - 25, x1 + 30, y1 + 25))
    tl, tr, br, bl = result.corners
    assert tl[0] < tr[0] and bl[0] < br[0]  # left points left of right points
    assert tl[1] < bl[1] and tr[1] < br[1]  # top points above bottom points


def test_degenerate_rect_returns_fallback():
    frame = np.full((50, 50, 3), 0.5, dtype=np.float32)
    result = detect_chart_quad(frame, (10, 10, 12, 11))
    assert not result.detected
