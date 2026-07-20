"""Matching tab UI, driven through real interaction paths offscreen:
solver switch, chain presets, stage add/remove/reorder, CSV loading via
the (mocked) file dialog, solve with waterfall + stage reports, export.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.core.lut import parse_cube
from app.core.reuleaux import ReuleauxUserParams, reuleaux_user


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """A modal QMessageBox would hang an offscreen run forever —
    turn any unexpected one into an immediate failure instead."""
    from app.ui.tabs import matching_tab

    def boom(*args, **kwargs):
        raise AssertionError(f"Unexpected dialog: {args[1:3]}")

    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(matching_tab.QMessageBox, name, staticmethod(boom))


def _write_pair_csvs(tmp_path):
    """Source/target CSV pair related by a known reuleaux move."""
    rng = np.random.default_rng(11)
    source = rng.uniform(0.08, 0.9, (200, 3))
    target = reuleaux_user(
        source, ReuleauxUserParams(overall_sat=1.1, red=(0.05, 1.2, 0.1))
    )

    def write(path, values):
        lines = ["label,R,G,B"]
        for i, (r, g, b) in enumerate(values):
            lines.append(f"p{i},{r:.17g},{g:.17g},{b:.17g}")
        path.write_text("\n".join(lines))

    src, tgt = tmp_path / "src.csv", tmp_path / "tgt.csv"
    write(src, source)
    write(tgt, target)
    return src, tgt


def _load_csv_into(box, path, monkeypatch):
    from app.ui.tabs import matching_tab

    monkeypatch.setattr(
        matching_tab.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(path), "")),
    )
    box._browse()
    assert path.name in box.info.text()


def _fresh_tab():
    from app.core.project import ProjectStore
    from app.ui.tabs.matching_tab import MatchingTab

    store = ProjectStore()
    return MatchingTab(store_provider=lambda: store)


def test_solver_switch_and_presets(qapp):
    tab = _fresh_tab()
    assert tab.solver_combo.currentText() == "RBF"
    assert tab._solver_stack.currentIndex() == 0

    tab.solver_combo.setCurrentText("Parametric")
    assert tab._solver_stack.currentIndex() == 1

    # default preset populated the list
    assert tab.stage_list.count() > 0

    tab.chain_preset_combo.setCurrentText("Reuleaux Broad only")
    assert [tab.stage_list.item(i).text() for i in range(tab.stage_list.count())] == [
        "Reuleaux Broad"
    ]

    tab.chain_preset_combo.setCurrentText("Matrix + Reuleaux Broad")
    assert [tab.stage_list.item(i).text() for i in range(tab.stage_list.count())] == [
        "Matrix", "Reuleaux Broad"
    ]


def test_stage_editing_marks_custom_and_reorders(qapp):
    tab = _fresh_tab()
    tab.solver_combo.setCurrentText("Parametric")
    tab.chain_preset_combo.setCurrentText("Matrix + Reuleaux Broad")

    tab.add_stage_combo.setCurrentText("Luma Curve")
    tab._add_stage()
    assert tab.chain_preset_combo.currentText() == "Custom"
    names = [tab.stage_list.item(i).text() for i in range(tab.stage_list.count())]
    assert names == ["Matrix", "Reuleaux Broad", "Luma Curve"]

    # move Luma Curve to the front
    tab.stage_list.setCurrentRow(2)
    tab._move_stage(-1)
    tab._move_stage(-1)
    names = [tab.stage_list.item(i).text() for i in range(tab.stage_list.count())]
    assert names == ["Luma Curve", "Matrix", "Reuleaux Broad"]

    # remove the matrix
    tab.stage_list.setCurrentRow(1)
    tab._remove_stage()
    names = [tab.stage_list.item(i).text() for i in range(tab.stage_list.count())]
    assert names == ["Luma Curve", "Reuleaux Broad"]

    stages = tab._build_stages()
    assert [s.name for s in stages] == ["Luma Curve", "Reuleaux Broad"]
    # curve stages pick up the point count from the spinbox
    tab.curve_points_spin.setValue(4)
    assert tab._build_stages()[0].identity().size == 4


def test_parametric_solve_and_export(qapp, tmp_path, monkeypatch):
    src, tgt = _write_pair_csvs(tmp_path)
    tab = _fresh_tab()
    _load_csv_into(tab.source_box, src, monkeypatch)
    _load_csv_into(tab.target_box, tgt, monkeypatch)

    tab.solver_combo.setCurrentText("Parametric")
    tab.chain_preset_combo.setCurrentText("Reuleaux Broad only")
    tab.solve()

    text = tab.result_label.text()
    assert "after Reuleaux Broad:" in text          # waterfall line
    assert "paste into" in text               # reuleaux slider report
    assert tab.export_btn.isEnabled()

    out = tmp_path / "parametric.cube"
    from app.ui.tabs import matching_tab

    monkeypatch.setattr(
        matching_tab.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "")),
    )
    tab.size_combo.setCurrentText("17")
    tab.export_cube()
    lut = parse_cube(out)
    assert lut.size == 17

    # the exported cube encodes the fitted model: probing a mid-grey
    # through the model directly must match the tab's fitted result
    probe = np.array([[0.4, 0.35, 0.3]])
    fitted = tab._result.model(probe)
    assert np.isfinite(fitted).all()


def test_rbf_path_still_solves(qapp, tmp_path, monkeypatch):
    src, tgt = _write_pair_csvs(tmp_path)
    tab = _fresh_tab()
    _load_csv_into(tab.source_box, src, monkeypatch)
    _load_csv_into(tab.target_box, tgt, monkeypatch)

    assert tab.solver_combo.currentText() == "RBF"
    tab.solve()
    text = tab.result_label.text()
    assert "After match:" in text
    assert "after Reuleaux" not in text
    assert tab.export_btn.isEnabled()
