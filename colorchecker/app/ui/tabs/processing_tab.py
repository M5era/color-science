"""Processing (readout) tab — the main working surface.

Phase 0: layout placeholder. Phases 1-5 fill in canvas, sidebar,
patch table, session list and export controls.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ProcessingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        placeholder = QLabel("Processing — canvas coming in Phase 1")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)
