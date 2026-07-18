"""Matching tab — placeholder until its phase lands."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class MatchingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        placeholder = QLabel("Matching — coming later")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)
