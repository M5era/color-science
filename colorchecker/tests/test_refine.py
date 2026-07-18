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


def _sg_like_frame():
    """Physical Digital SG layout: 10x14 patch mosaic where the outer ring
    is achromatic (grays) and the inner 8x12 field is colored."""
    rng = np.random.default_rng(21)
    frame = np.full((500, 700, 3), 0.2, dtype=np.float32)
    x0, y0 = 80, 70
    pitch = 38
    for r in range(10):
        for c in range(14):
            if r in (0, 9) or c in (0, 13):
                value = np.full(3, rng.uniform(0.1, 0.55), dtype=np.float32)  # gray ring
            else:
                value = rng.uniform(0.1, 0.55, 3).astype(np.float32)
            ys, xs = y0 + r * pitch, x0 + c * pitch
            frame[ys + 3 : ys + pitch - 3, xs + 3 : xs + pitch - 3] = value
    full_field = (x0, y0, x0 + 14 * pitch, y0 + 10 * pitch)
    inner_field = (x0 + pitch, y0 + pitch, x0 + 13 * pitch, y0 + 9 * pitch)
    return frame, full_field, inner_field


def test_align_grid_finds_inner_8x12_window_of_sg():
    from app.core.refine import align_grid

    frame, full, inner = _sg_like_frame()
    detected = [[full[0], full[1]], [full[2], full[1]], [full[2], full[3]], [full[0], full[3]]]
    result = align_grid(frame, detected, rows=8, cols=12)
    corners = np.asarray(result.corners)
    expected = np.array(
        [[inner[0], inner[1]], [inner[2], inner[1]], [inner[2], inner[3]], [inner[0], inner[3]]]
    )
    # Corners of the aligned window within ~a third of a patch of truth.
    assert np.abs(corners - expected).max() < 13
