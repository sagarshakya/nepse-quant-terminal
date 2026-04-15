"""Bottom console log tail. Simple for MVP; tabs added later."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QFrame, QPlainTextEdit, QVBoxLayout, QWidget

from apps.desktop.theme import BG, BORDER, PANE, TEXT, TEXT_SECONDARY, mono_font


class Console(QFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"Console {{ background: {PANE}; border-top: 1px solid {BORDER}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setFont(mono_font(11))
        self.view.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {BG}; color: {TEXT_SECONDARY};
                border: none; padding: 4px 8px;
            }}
        """)
        self.view.setFrameStyle(QFrame.NoFrame)
        lay.addWidget(self.view)

    def log(self, source: str, message: str, level: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        color = {
            "info": TEXT_SECONDARY,
            "warn": "#E0A23A",
            "err":  "#E5484D",
            "ok":   "#2AC27D",
        }.get(level, TEXT_SECONDARY)
        line = f"<span style='color:#5B626C'>{ts}</span>  "\
               f"<span style='color:#8A93A0'>{source:<8}</span>  "\
               f"<span style='color:{color}'>{message}</span>"
        self.view.appendHtml(line)
        c = self.view.textCursor()
        c.movePosition(QTextCursor.End)
        self.view.setTextCursor(c)
