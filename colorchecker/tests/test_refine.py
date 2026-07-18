"""Grid refinement finds the true patch-field inset from the outer quad."""

import numpy as np

from app.core.refine import refine_margins


def _chart_with_inset(margin_x_pct=8.0, margin_y_pct=11.0):
    """Chart occupying a known quad, patch field inset by known margins —
    like detecting a chart's outer plastic edge."""
    rng = np.random.default_rng(5)
    frame = np.full((400, 560, 3), 0.22, dtype=np.float32)
    x0, y0, x1, y1 = 60, 50, 500, 350
    frame[y0:y1, x0:x1] = 0.06  # chart body / border

    width, height = x1 - x0, y1 - y0
    mx = int(width * margin_x_pct / 100)
    my = int(height * margin_y_pct / 100)
    inner = frame[y0 + my : y1 - my, x0 + mx : x1 - mx]
    rows, cols = 8, 12
    ph, pw = inner.shape[0] / rows, inner.shape[1] / cols
    for r in range(rows):
        for c in range(cols):
            value = rng.uniform(0.1, 0.6, 3).astype(np.float32)
            ys, ye = int(r * ph + 2), int((r + 1) * ph - 2)
            xs, xe = int(c * pw + 2), int((c + 1) * pw - 2)
            inner[ys:ye, xs:xe] = value
    return frame, (x0, y0, x1, y1)


def test_recovers_known_margins():
    frame, (x0, y0, x1, y1) = _chart_with_inset(margin_x_pct=8.0, margin_y_pct=11.0)
    corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    result = refine_margins(frame, corners, rows=8, cols=12)
    assert abs(result.margin_x - 8.0) < 1.5
    assert abs(result.margin_y - 11.0) < 1.5


def test_zero_margin_chart_stays_near_zero():
    frame, (x0, y0, x1, y1) = _chart_with_inset(margin_x_pct=0.5, margin_y_pct=0.5)
    corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    result = refine_margins(frame, corners, rows=8, cols=12)
    assert result.margin_x < 3.0
    assert result.margin_y < 3.0
