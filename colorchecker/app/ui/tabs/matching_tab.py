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
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.match import (
    MatchResult,
    load_patch_csv,
    session_patch_rows,
    solve_match,
    write_cube,
)


class _PatchSource(QGroupBox):
    """One side of the match: current session or a CSV file."""

    def __init__(self, title: str, store_provider, default_session: bool):
        super().__init__(title)
        self._store_provider = store_provider
        self._csv_values: np.ndarray | None = None
        self._csv_labels: list[str] = []

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
        self.csv_radio.setChecked(True)
        self._csv_name = Path(path).name
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

        params = QGroupBox("Model")
        grid = QGridLayout(params)
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
        row += 1

        grid.addWidget(QLabel("Strength (%)"), row, 0)
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(0, 100)
        self.strength_spin.setValue(100)
        self.strength_spin.setToolTip("Blend the whole match against the original")
        grid.addWidget(self.strength_spin, row, 1)
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

    # ------------------------------------------------------------ actions

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
        try:
            result = solve_match(
                source,
                target,
                use_matrix=self.matrix_check.isChecked(),
                layers=self.layers_spin.value(),
                smoothing=self.smoothness_spin.value(),
                strength=self.strength_spin.value() / 100.0,
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot solve", str(exc))
            return

        self._result = result
        self.export_btn.setEnabled(True)

        parts = [
            f"Pairs used: {result.pairs_used}"
            + (f" ({result.pairs_dropped} dropped: missing values)" if result.pairs_dropped else ""),
            f"Error before: {result.error_before:.5f}",
        ]
        if result.error_matrix is not None:
            parts.append(f"After matrix: {result.error_matrix:.5f}")
        parts.append(
            f"After match: {result.error_after:.5f} (worst patch {result.error_after_max:.5f})"
        )

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
