"""PaneHeader — consistent 26px title bar with uppercase label, link badge,
and an optional actions area on the right."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from apps.desktop.theme import (
    PANE, BORDER, TEXT, TEXT_SECONDARY, ACCENT, ui_font, mono_font,
)


class LinkDot(QFrame):
    def __init__(self, color: str = ACCENT):
        super().__init__()
        self._color = color
        self.setFixedSize(8, 8)

    def set_color(self, color: str):
        self._color = color
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(Qt.NoPen))
        p.setBrush(QBrush(QColor(self._color)))
        p.drawEllipse(0, 0, 8, 8)


class PaneHeader(QFrame):
    """Thin title bar used on every pane for a consistent visual language."""

    def __init__(self, title: str, link_color: Optional[str] = None, subtitle: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.setStyleSheet(
            f"PaneHeader {{ background: {PANE}; border-bottom: 1px solid {BORDER}; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(8)

        if link_color:
            self._dot = LinkDot(link_color)
            lay.addWidget(self._dot, 0, Qt.AlignVCenter)
        else:
            self._dot = None

        self.title = QLabel(title.upper())
        self.title.setFont(ui_font(11, 600))
        self.title.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 1.2px;")
        lay.addWidget(self.title)

        self.subtitle = QLabel(subtitle)
        self.subtitle.setFont(mono_font(11))
        self.subtitle.setStyleSheet(f"color: {TEXT_SECONDARY};")
        lay.addWidget(self.subtitle)

        lay.addStretch(1)

        # Actions area on the right
        self._actions = QHBoxLayout()
        self._actions.setSpacing(6)
        lay.addLayout(self._actions)

    def set_subtitle(self, text: str):
        self.subtitle.setText(text)

    def add_action(self, widget: QWidget):
        self._actions.addWidget(widget)

    def set_link_color(self, color: str):
        if self._dot is not None:
            self._dot.set_color(color)
