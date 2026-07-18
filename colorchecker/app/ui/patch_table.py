"""Bottom patch table: sampled values in chart layout, cells tinted
with the patch color (display-mapped, clamped — tint is cosmetic only)."""

import numpy as np
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

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
