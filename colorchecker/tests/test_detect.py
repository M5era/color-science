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


def _realistic_log_frame():
    """The failure case from real footage: chart border blends into a dark
    shelf, chart sits OFF-CENTER in the drawn box, clutter elsewhere."""
    rng = np.random.default_rng(9)
    frame = np.full((360, 520, 3), 0.21, dtype=np.float32)
    frame += rng.normal(0, 0.004, frame.shape).astype(np.float32)
    frame[250:, :] = 0.07  # dark shelf/table across the bottom
    # Chart body barely darker than shelf, top half against the wall.
    x0, y0, x1, y1 = 120, 90, 340, 260
    frame[y0:y1, x0:x1] = 0.065
    # Patch mosaic with visible gaps (the detectable signature).
    inner = frame[y0 + 10 : y1 - 10, x0 + 10 : x1 - 10]
    rows, cols = 8, 12
    ph, pw = inner.shape[0] / rows, inner.shape[1] / cols
    for r in range(rows):
        for c in range(cols):
            value = rng.uniform(0.12, 0.5, 3).astype(np.float32)
            ys, ye = int(r * ph) + 2, int((r + 1) * ph) - 2
            xs, xe = int(c * pw) + 2, int((c + 1) * pw) - 2
            inner[ys:ye, xs:xe] = value
    # Clutter: bright sticky notes and a second chart edge outside the box.
    frame[20:45, 60:150] = [0.5, 0.55, 0.3]
    frame[300:340, 420:500] = [0.55, 0.3, 0.4]
    return frame, (x0 + 10, y0 + 10, x1 - 10, y1 - 10)


def test_offcenter_chart_low_border_contrast():
    frame, (fx0, fy0, fx1, fy1) = _realistic_log_frame()
    # Loose box, chart well off-center: lots of wall above, shelf below.
    result = detect_chart_quad(frame, (60, 20, 420, 340))
    assert result.detected
    corners = np.asarray(result.corners)
    expected = np.array([[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]])
    # Patch-field quad within ~8 px of the mosaic bounds on every corner.
    assert np.abs(corners - expected).max() < 8
