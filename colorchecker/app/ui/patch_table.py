"""Bottom results panel: sampled values in chart layout, cells tinted
with the patch color (display-mapped, clamped — tint is cosmetic only).

PatchTablePanel shows every overlay that has results side by side —
the chart grid and a light-source square appear together instead of
being filtered by the sidebar's overlay selection."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.sampler import PatchSample


class PatchTable(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setMinimumHeight(180)
        self.setMaximumHeight(320)

    def show_samples(self, samples: list[PatchSample], rows: int, cols: int) -> None:
        self.setRowCount(rows)
        self.setColumnCount(cols)
        self.setHorizontalHeaderLabels([str(c + 1) for c in range(cols)])
        self.setVerticalHeaderLabels([str(r + 1) for r in range(rows)])

        for sample in samples:
            r, g, b = sample.rgb
            if sample.pixel_count == 0 or any(np.isnan(v) for v in sample.rgb):
                item = QTableWidgetItem("—")
                item.setBackground(QBrush(QColor(60, 60, 60)))
                item.setForeground(QBrush(QColor(160, 160, 160)))
            else:
                item = QTableWidgetItem(f"{r:.6f}\n{g:.6f}\n{b:.6f}")
                tint = QColor(
                    int(np.clip(r, 0, 1) * 255),
                    int(np.clip(g, 0, 1) * 255),
                    int(np.clip(b, 0, 1) * 255),
                )
                item.setBackground(QBrush(tint))
                luminance = 0.2126 * np.clip(r, 0, 1) + 0.7152 * np.clip(g, 0, 1) + 0.0722 * np.clip(b, 0, 1)
                text = QColor(0, 0, 0) if luminance > 0.35 else QColor(255, 255, 255)
                item.setForeground(QBrush(text))
            self.setItem(sample.row - 1, sample.col - 1, item)

        self.resizeRowsToContents()
        self.resizeColumnsToContents()


def _sample_from_dict(data: dict) -> PatchSample:
    return PatchSample(
        row=data["row"],
        col=data["col"],
        rgb=tuple(data["rgb"]),
        pixel_count=data.get("pixel_count", 0),
    )


class PatchTablePanel(QWidget):
    """All overlays' results at once: one titled table per overlay,
    laid out horizontally; the widest grid gets the most space."""

    def __init__(self):
        super().__init__()
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 0)
        self._layout.setSpacing(8)
        self.setMaximumHeight(340)
        self.hide()

    def show_results(self, results: list[dict]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        # Group rows by overlay, preserving order of first appearance.
        order: list[str] = []
        grouped: dict[str, list[dict]] = {}
        for row in results:
            name = row.get("overlay", "Overlay 1")
            if name not in grouped:
                grouped[name] = []
                order.append(name)
            grouped[name].append(row)

        if not order:
            self.hide()
            return

        for name in order:
            rows_data = grouped[name]
            n_rows = max(r["row"] for r in rows_data)
            n_cols = max(r["col"] for r in rows_data)
            kind = rows_data[0].get("kind", "reflective")

            box = QWidget()
            vbox = QVBoxLayout(box)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(2)
            title = QLabel(f"{name} — {kind}")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            vbox.addWidget(title)
            table = PatchTable()
            table.show_samples([_sample_from_dict(r) for r in rows_data], n_rows, n_cols)
            vbox.addWidget(table)
            self._layout.addWidget(box, stretch=n_cols)

        self.show()
