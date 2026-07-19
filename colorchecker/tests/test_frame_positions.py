"""Per-frame overlay position overrides — the v2 gauntlet.

Runs the real UI offscreen, driving actual interaction paths: detect via
the canvas signal, preset changes through the sidebar combo, overlay
add/remove cycles (the suspected v1 breaker: repeated display names must
not cross-contaminate positions, hence UID keying).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import tifffile
from PySide6.QtWidgets import QApplication

from app.core.project import ProjectStore


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _patch_frame(shift=0):
    """2x3 uniform patch chart at x=60+shift; values 0.1..0.6."""
    frame = np.full((200, 320, 3), 0.05, dtype=np.float32)
    x0, y0 = 60 + shift, 50
    for r in range(2):
        for c in range(3):
            value = np.float32(0.1 + 0.1 * (r * 3 + c))
            frame[y0 + r * 40 : y0 + (r + 1) * 40, x0 + c * 40 : x0 + (c + 1) * 40] = value
    return frame, x0, y0


def _configure_chart_overlay(pt, x0, y0):
    item = pt._active_item()
    overlay = item.overlay
    overlay.rows, overlay.cols = 2, 3
    overlay.margin_x = overlay.margin_y = 0.0
    overlay.patch_size = 50.0
    overlay.corners = [[x0, y0], [x0 + 120, y0], [x0 + 120, y0 + 80], [x0, y0 + 80]]
    item.model_changed()
    pt._persist_corners(overlay.uid)
    return overlay


EXPECTED = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def _reds(entry, overlay_name="Overlay 1"):
    return [round(r["rgb"][0], 6) for r in entry.patch_results if r["overlay"] == overlay_name]


def test_override_gauntlet(qapp, tmp_path):
    from app.ui.main_window import MainWindow, Tab

    fa, ax, ay = _patch_frame(0)
    fb, bx, by = _patch_frame(24)  # the bumped shot
    tifffile.imwrite(tmp_path / "A_EV0.tif", fa, photometric="rgb")
    tifffile.imwrite(tmp_path / "B_EV+1.tif", fb, photometric="rgb")

    window = MainWindow()
    window.show()
    qapp.processEvents()
    pt = window._tabs[Tab.PROCESSING]
    for p in sorted(tmp_path.glob("*.tif")):
        pt._ensure_entry(p)
    pt._refresh_session_list()
    pt.open_image(tmp_path / "A_EV0.tif")
    qapp.processEvents()

    # --- add/remove cycle FIRST (the v1 breaker) -----------------------
    pt._add_overlay()          # "Overlay 2"
    pt._remove_active_overlay()
    pt._add_overlay()          # re-add: same display name, DIFFERENT uid
    second = pt._active_item().overlay
    pt._remove_active_overlay()
    pt._on_overlay_selected(0)

    chart = _configure_chart_overlay(pt, ax, ay)
    shared_before = [list(c) for c in pt._shared_corners[chart.uid]]

    # --- override on the bumped frame via the real toggle path ---------
    pt.open_image(tmp_path / "B_EV+1.tif")
    qapp.processEvents()
    pt.sidebar.local_pos_check.setChecked(True)   # real checkbox path
    qapp.processEvents()
    item = pt._active_item()
    item.overlay.corners = [[bx, by], [bx + 120, by], [bx + 120, by + 80], [bx, by + 80]]
    item.model_changed()
    pt._persist_corners(chart.uid)

    entry_b = pt.store.images[1]
    assert chart.uid in entry_b.overlay_overrides
    # Shared position untouched by the override edit:
    assert pt._shared_corners[chart.uid] == shared_before

    # --- frame A still shows shared; checkbox reflects per-frame state -
    pt.open_image(tmp_path / "A_EV0.tif")
    qapp.processEvents()
    assert pt._active_item().overlay.corners[0] == [ax, ay]
    assert not pt.sidebar.local_pos_check.isChecked()
    pt.open_image(tmp_path / "B_EV+1.tif")
    qapp.processEvents()
    assert pt.sidebar.local_pos_check.isChecked()
    assert pt._active_item().overlay.corners[0] == [bx, by]

    # --- Process All samples both frames bit-exact ---------------------
    pt.process_all()
    qapp.processEvents()
    for entry in pt.store.images:
        assert _reds(entry) == EXPECTED, entry.label

    # --- overrides survive project save/load ---------------------------
    project = tmp_path / "p.ccproj.json"
    pt.store.save(project)
    loaded = ProjectStore.load(project)
    assert chart.uid in loaded.images[1].overlay_overrides

    # --- untick reverts to shared --------------------------------------
    pt.sidebar.local_pos_check.setChecked(False)
    qapp.processEvents()
    assert pt._active_item().overlay.corners[0] == [ax, ay]
    assert chart.uid not in pt.store.images[1].overlay_overrides


def test_detect_unaffected_by_feature(qapp, tmp_path):
    """Detect via the real canvas signal, before and after add/remove
    cycles and with an override active on another frame."""
    from tests.test_detect import _realistic_log_frame
    from app.ui.main_window import MainWindow, Tab

    frame, field = _realistic_log_frame()
    tifffile.imwrite(tmp_path / "A_EV0.tif", frame, photometric="rgb")
    tifffile.imwrite(tmp_path / "B_EV+1.tif", frame, photometric="rgb")

    window = MainWindow()
    window.show()
    qapp.processEvents()
    pt = window._tabs[Tab.PROCESSING]
    for p in sorted(tmp_path.glob("*.tif")):
        pt._ensure_entry(p)
    pt._refresh_session_list()
    pt.open_image(tmp_path / "A_EV0.tif")
    qapp.processEvents()

    # add/remove churn, then detect through the CANVAS SIGNAL
    pt._add_overlay(); pt._remove_active_overlay()
    pt._add_overlay(); pt._remove_active_overlay()
    pt._on_overlay_selected(0)
    pt.canvas.rectSelected.emit(60.0, 20.0, 420.0, 340.0)
    qapp.processEvents()

    corners = np.asarray(pt._active_item().overlay.corners)
    expected = np.array(
        [[field[0], field[1]], [field[2], field[1]],
         [field[2], field[3]], [field[0], field[3]]]
    )
    assert np.abs(corners - expected).max() < 8

    # Detection carried to the other frame via the shared position
    pt.open_image(tmp_path / "B_EV+1.tif")
    qapp.processEvents()
    np.testing.assert_allclose(pt._active_item().overlay.corners, corners)

    # Detect again on B with a local override: A must NOT move
    pt.sidebar.local_pos_check.setChecked(True)
    qapp.processEvents()
    pt.canvas.rectSelected.emit(60.0, 20.0, 420.0, 340.0)
    qapp.processEvents()
    pt.open_image(tmp_path / "A_EV0.tif")
    qapp.processEvents()
    np.testing.assert_allclose(pt._active_item().overlay.corners, corners)
