"""Matching tab: fit source patches to target patches, export a .cube.

Source and target each come from the current session or a CSV file.
Model: optional 3x3 matrix pre-fit + hierarchical RBF on the residual,
with a global strength blend. Rows must pair up in the same order —
which they do automatically when both sides were exported/processed
with the same overlay setup and session ordering.
"""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.lut import CubeLUT, parse_cube
from app.core.match import (
    MatchResult,
    load_patch_csv,
    session_patch_rows,
    solve_match,
    write_cube,
)
from app.core.parametric import solve_parametric
from app.core.stages import CHAIN_PRESETS, STAGE_POOL, LumaCurveStage, RGBCurvesStage


class _PatchSource(QGroupBox):
    """One side of the match: current session or a CSV file."""

    def __init__(self, title: str, store_provider, default_session: bool):
        super().__init__(title)
        self._store_provider = store_provider
        self._csv_values: np.ndarray | None = None
        self._csv_labels: list[str] = []
        self._csv_name: str = ""

        layout = QVBoxLayout(self)
        self.session_radio = QRadioButton("Current session (Processing tab)")
        self.csv_radio = QRadioButton("CSV file")
        (self.session_radio if default_session else self.csv_radio).setChecked(True)
        layout.addWidget(self.session_radio)

        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_radio)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        csv_row.addWidget(browse)
        layout.addLayout(csv_row)

        self.info = QLabel("—")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        self.session_radio.toggled.connect(lambda _: self.refresh())
        self.refresh()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load patch CSV", "", "CSV / text files (*.csv *.txt);;All files (*)"
        )
        if not path:
            return
        try:
            self._csv_values, self._csv_labels = load_patch_csv(path)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot read file", str(exc))
            return
        # Name must be recorded BEFORE toggling the radio: the toggle
        # signal triggers refresh(), which displays the name.
        self._csv_name = Path(path).name
        self.csv_radio.setChecked(True)
        self.refresh()

    def values(self) -> tuple[np.ndarray, list[str]]:
        if self.session_radio.isChecked():
            return session_patch_rows(self._store_provider())
        if self._csv_values is None:
            return np.empty((0, 3)), []
        return self._csv_values, self._csv_labels

    def refresh(self) -> None:
        values, _ = self.values()
        if self.session_radio.isChecked():
            self.info.setText(f"{len(values)} patch rows from the session")
        elif self._csv_values is None:
            self.info.setText("No file loaded")
        else:
            self.info.setText(f"{len(values)} patch rows from {self._csv_name}")


class MatchingTab(QWidget):
    def __init__(self, store_provider=None):
        super().__init__()
        self._store_provider = store_provider or (lambda: None)
        self._result: MatchResult | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left = QVBoxLayout()
        self.source_box = _PatchSource("Source (footage to correct)",
                                       self._store_provider, default_session=True)
        self.target_box = _PatchSource("Target (reference to match)",
                                       self._store_provider, default_session=False)
        left.addWidget(self.source_box)
        left.addWidget(self.target_box)
        left.addStretch(1)
        root.addLayout(left, stretch=1)

        right = QVBoxLayout()

        match_type = QGroupBox("Match type")
        mt_layout = QVBoxLayout(match_type)
        self.scene_radio = QRadioButton("Scene-referred (log → log)")
        self.scene_radio.setChecked(True)
        self.display_radio = QRadioButton("Display-referred (through a DRT)")
        self.display_radio.setToolTip(
            "Target is a display-referred scan (e.g. slide film). A fixed\n"
            "DRT carries the contrast; the match is solved underneath it\n"
            "and exports as a cube you stack BEFORE the DRT."
        )
        mt_layout.addWidget(self.scene_radio)
        mt_layout.addWidget(self.display_radio)

        drt_row = QHBoxLayout()
        drt_row.addWidget(QLabel("DRT:"))
        self.drt_combo = QComboBox()
        self.drt_combo.setEnabled(False)
        drt_row.addWidget(self.drt_combo, stretch=1)
        self.drt_load_btn = QPushButton("Load DRT…")
        self.drt_load_btn.setEnabled(False)
        self.drt_load_btn.clicked.connect(self._load_drt)
        drt_row.addWidget(self.drt_load_btn)
        mt_layout.addLayout(drt_row)
        self._drts: list[tuple[str, CubeLUT]] = []
        self.display_radio.toggled.connect(self._match_type_changed)
        right.addWidget(match_type)

        params = QGroupBox("Model")
        params_box = QVBoxLayout(params)

        solver_row = QHBoxLayout()
        solver_row.addWidget(QLabel("Solver:"))
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["RBF", "Parametric"])
        self.solver_combo.setToolTip(
            "RBF: best raw accuracy, exports a 3D cube.\n"
            "Parametric: chain of interpretable stages (curves, matrix,\n"
            "Reuleaux) — exports slider values and 1D curves you can\n"
            "rebuild natively in Resolve."
        )
        solver_row.addWidget(self.solver_combo, stretch=1)
        params_box.addLayout(solver_row)

        self._solver_stack = QStackedWidget()
        params_box.addWidget(self._solver_stack)

        # -------- RBF page --------
        rbf_page = QWidget()
        grid = QGridLayout(rbf_page)
        grid.setContentsMargins(0, 4, 0, 0)
        row = 0

        self.matrix_check = QCheckBox("3×3 matrix pre-fit")
        self.matrix_check.setChecked(True)
        self.matrix_check.setToolTip(
            "Fit a robust linear matrix first; the RBF only corrects the rest"
        )
        grid.addWidget(self.matrix_check, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("Smoothness"), row, 0)
        self.smoothness_spin = QDoubleSpinBox()
        self.smoothness_spin.setRange(0.0, 0.5)
        self.smoothness_spin.setDecimals(4)
        self.smoothness_spin.setSingleStep(0.001)
        self.smoothness_spin.setValue(0.001)
        self.smoothness_spin.setToolTip(
            "Low: trust every patch exactly. Higher: smoother, tolerates noise"
        )
        grid.addWidget(self.smoothness_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Detail layers"), row, 0)
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(0, 10)
        self.layers_spin.setValue(10)
        self.layers_spin.setToolTip("0 = matrix only; 10 = finest RBF detail")
        grid.addWidget(self.layers_spin, row, 1)
        self._solver_stack.addWidget(rbf_page)

        # -------- Parametric page --------
        para_page = QWidget()
        para_box = QVBoxLayout(para_page)
        para_box.setContentsMargins(0, 4, 0, 0)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Chain:"))
        self.chain_preset_combo = QComboBox()
        self.chain_preset_combo.addItems(list(CHAIN_PRESETS) + ["Custom"])
        self.chain_preset_combo.currentTextChanged.connect(self._apply_chain_preset)
        preset_row.addWidget(self.chain_preset_combo, stretch=1)
        para_box.addLayout(preset_row)

        self.stage_list = QListWidget()
        self.stage_list.setToolTip("Stages run top to bottom; only listed stages are solved")
        para_box.addWidget(self.stage_list)

        stage_buttons = QHBoxLayout()
        self.add_stage_combo = QComboBox()
        self.add_stage_combo.addItems(list(STAGE_POOL))
        stage_buttons.addWidget(self.add_stage_combo)
        for text, slot in (
            ("Add", self._add_stage), ("Remove", self._remove_stage),
            ("↑", lambda: self._move_stage(-1)), ("↓", lambda: self._move_stage(+1)),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            stage_buttons.addWidget(btn)
        para_box.addLayout(stage_buttons)

        points_row = QHBoxLayout()
        points_row.addWidget(QLabel("Curve points"))
        self.curve_points_spin = QSpinBox()
        self.curve_points_spin.setRange(3, 12)
        self.curve_points_spin.setValue(6)
        points_row.addWidget(self.curve_points_spin)
        points_row.addStretch(1)
        para_box.addLayout(points_row)

        from app.core.backprop import torch_available

        self.backprop_check = QCheckBox("Backprop refine (PyTorch)")
        if torch_available():
            self.backprop_check.setToolTip(
                "Gradient optimization with multi-restart placement of "
                "Reuleaux Fine zones — finds zones anywhere on the hue "
                "wheel instead of only near their starting position. "
                "Slower per solve."
            )
        else:
            self.backprop_check.setEnabled(False)
            self.backprop_check.setToolTip(
                "PyTorch is not installed — enable with: "
                "python3 -m pip install torch"
            )
        para_box.addWidget(self.backprop_check)
        self._solver_stack.addWidget(para_page)

        self.solver_combo.currentIndexChanged.connect(self._solver_stack.setCurrentIndex)
        self._apply_chain_preset(self.chain_preset_combo.currentText())

        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Strength (%)"))
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(0, 100)
        self.strength_spin.setValue(100)
        self.strength_spin.setToolTip("Blend the whole match against the original")
        strength_row.addWidget(self.strength_spin)
        strength_row.addStretch(1)
        params_box.addLayout(strength_row)
        right.addWidget(params)

        export = QGroupBox("LUT export")
        egrid = QGridLayout(export)
        egrid.addWidget(QLabel("Size"), 0, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["17", "33", "65"])
        self.size_combo.setCurrentText("33")
        egrid.addWidget(self.size_combo, 0, 1)
        egrid.addWidget(QLabel("Domain min"), 1, 0)
        self.domain_min_spin = QDoubleSpinBox()
        self.domain_min_spin.setRange(-4.0, 4.0)
        self.domain_min_spin.setDecimals(2)
        self.domain_min_spin.setValue(0.0)
        egrid.addWidget(self.domain_min_spin, 1, 1)
        egrid.addWidget(QLabel("Domain max"), 2, 0)
        self.domain_max_spin = QDoubleSpinBox()
        self.domain_max_spin.setRange(0.1, 16.0)
        self.domain_max_spin.setDecimals(2)
        self.domain_max_spin.setValue(1.0)
        self.domain_max_spin.setToolTip(
            "Raise above 1.0 if your data (e.g. light sources) exceeds 1.0"
        )
        egrid.addWidget(self.domain_max_spin, 2, 1)
        right.addWidget(export)

        solve_btn = QPushButton("Solve Match")
        solve_btn.clicked.connect(self.solve)
        right.addWidget(solve_btn)

        self.result_label = QLabel("Not solved yet")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        right.addWidget(self.result_label)

        self.export_btn = QPushButton("Export .cube")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_cube)
        right.addWidget(self.export_btn)
        right.addStretch(1)
        root.addLayout(right, stretch=1)

    # ---------------------------------------------------- stage chain

    def _apply_chain_preset(self, preset: str) -> None:
        if preset in CHAIN_PRESETS:
            self.stage_list.clear()
            self.stage_list.addItems(CHAIN_PRESETS[preset])

    def _mark_custom(self) -> None:
        self.chain_preset_combo.blockSignals(True)
        self.chain_preset_combo.setCurrentText("Custom")
        self.chain_preset_combo.blockSignals(False)

    def _add_stage(self) -> None:
        self.stage_list.addItem(self.add_stage_combo.currentText())
        self._mark_custom()

    def _remove_stage(self) -> None:
        row = self.stage_list.currentRow()
        if row >= 0:
            self.stage_list.takeItem(row)
            self._mark_custom()

    def _move_stage(self, delta: int) -> None:
        row = self.stage_list.currentRow()
        target = row + delta
        if row < 0 or not (0 <= target < self.stage_list.count()):
            return
        item = self.stage_list.takeItem(row)
        self.stage_list.insertItem(target, item)
        self.stage_list.setCurrentRow(target)
        self._mark_custom()

    def _build_stages(self) -> list:
        stages = []
        points = self.curve_points_spin.value()
        for i in range(self.stage_list.count()):
            name = self.stage_list.item(i).text()
            cls = STAGE_POOL[name]
            if cls in (LumaCurveStage, RGBCurvesStage):
                stages.append(cls(points))
            else:
                stages.append(cls())
        return stages

    # ------------------------------------------------------------ actions

    def _match_type_changed(self, display_mode: bool) -> None:
        self.drt_combo.setEnabled(display_mode)
        self.drt_load_btn.setEnabled(display_mode)

    def _load_drt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load DRT", "", "Cube LUT (*.cube)"
        )
        if not path:
            return
        try:
            lut = parse_cube(path)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot load DRT", str(exc))
            return
        name = Path(path).name
        self._drts.append((name, lut))
        self.drt_combo.addItem(name)
        self.drt_combo.setCurrentIndex(len(self._drts) - 1)

    def _selected_drt(self) -> CubeLUT | None:
        if not self.display_radio.isChecked():
            return None
        idx = self.drt_combo.currentIndex()
        if idx < 0 or idx >= len(self._drts):
            return None
        return self._drts[idx][1]

    def showEvent(self, event):
        super().showEvent(event)
        self.source_box.refresh()
        self.target_box.refresh()

    def solve(self) -> None:
        source, source_labels = self.source_box.values()
        target, _ = self.target_box.values()
        if len(source) == 0 or len(target) == 0:
            QMessageBox.information(
                self, "Missing data",
                "Both source and target need patch rows (process the session "
                "or load a CSV on each side).",
            )
            return
        drt = self._selected_drt()
        if self.display_radio.isChecked() and drt is None:
            QMessageBox.information(
                self, "No DRT loaded",
                "Display-referred matching needs a DRT — click Load DRT…",
            )
            return
        parametric = self.solver_combo.currentText() == "Parametric"
        try:
            if parametric:
                result = solve_parametric(
                    source,
                    target,
                    stages=self._build_stages(),
                    strength=self.strength_spin.value() / 100.0,
                    output_transform=drt,
                    backend="torch" if self.backprop_check.isChecked() else "scipy",
                )
            else:
                result = solve_match(
                    source,
                    target,
                    use_matrix=self.matrix_check.isChecked(),
                    layers=self.layers_spin.value(),
                    smoothing=self.smoothness_spin.value(),
                    strength=self.strength_spin.value() / 100.0,
                    output_transform=drt,
                )
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot solve", str(exc))
            return

        self._result = result
        self.export_btn.setEnabled(True)

        parts = [
            f"Pairs used: {result.pairs_used}"
            + (f" ({result.pairs_dropped} dropped: missing values)" if result.pairs_dropped else "")
            + (
                f" ({result.pairs_unreachable} dropped: clipped by the DRT/stock)"
                if result.pairs_unreachable else ""
            ),
        ]
        if result.display_referred:
            parts.append("Errors measured through the DRT (what you'd see):")
        parts.append(f"Error before: {result.error_before:.5f}")
        if parametric:
            for (stage_name, err), (_, gain), label in zip(
                result.waterfall, result.stage_noise_gain,
                result.stage_labels,
            ):
                shown = stage_name if label == stage_name else f"{stage_name} — {label}"
                parts.append(
                    f"  after {shown}: {err:.5f}   "
                    f"[noise gain ×{gain['median']:.2f}, max ×{gain['max']:.2f}]"
                )
        elif result.error_matrix is not None:
            parts.append(f"After matrix: {result.error_matrix:.5f}")
        parts.append(
            f"After match: {result.error_after:.5f} (worst patch {result.error_after_max:.5f})"
        )
        if parametric and result.chain_noise_gain is not None:
            g = result.chain_noise_gain
            parts.append(
                f"Chain noise gain: ×{g['median']:.2f} median, "
                f"×{g['max']:.2f} max (≈1 = transparent; ≫1 amplifies noise)"
            )
        if result.display_referred:
            parts.append("Export = correction cube; apply it BEFORE the DRT node.")
        if parametric:
            parts.append("")
            parts.extend(result.stage_reports)

        valid_labels = [
            lbl for lbl, keep in zip(
                source_labels,
                ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1)),
            ) if keep
        ]
        if valid_labels and len(valid_labels) == len(result.per_patch_error):
            worst = np.argsort(result.per_patch_error)[-3:][::-1]
            parts.append("Worst patches:")
            for i in worst:
                parts.append(f"  {valid_labels[i]}: {result.per_patch_error[i]:.5f}")
        self.result_label.setText("\n".join(parts))

    def export_cube(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export LUT", "match.cube", "Cube LUT (*.cube)"
        )
        if not path:
            return
        try:
            write_cube(
                self._result.model,
                path,
                size=int(self.size_combo.currentText()),
                domain_min=self.domain_min_spin.value(),
                domain_max=self.domain_max_spin.value(),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
