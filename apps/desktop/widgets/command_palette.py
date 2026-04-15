"""Ctrl+K command palette with fuzzy matching over commands + tickers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog, QFrame, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout,
    QWidget, QLabel, QHBoxLayout,
)

from apps.desktop.theme import (
    ELEVATED, BG, BORDER, TEXT, TEXT_SECONDARY, ACCENT, ACCENT_SOFT,
    mono_font, ui_font,
)


@dataclass(slots=True)
class Command:
    label: str
    hint: str
    action: Callable[[], None]
    kind: str = "cmd"  # "cmd" | "ticker"


class CommandPalette(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setFixedWidth(560)
        self._commands: list[Command] = []
        self._build()

    def _build(self):
        self.setStyleSheet(
            f"QDialog {{ background: {ELEVATED}; border: 1px solid {BORDER}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Jump to ticker, run command…")
        self.input.setFont(mono_font(14))
        self.input.setFixedHeight(36)
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: {ELEVATED}; border: none; border-bottom: 1px solid {BORDER};
                color: {TEXT}; padding: 6px 12px;
            }}
        """)
        self.input.textChanged.connect(self._refilter)
        self.input.returnPressed.connect(self._activate_current)

        self.list = QListWidget()
        self.list.setFont(ui_font(12))
        self.list.setStyleSheet(f"""
            QListWidget {{ background: {ELEVATED}; border: none; outline: 0; }}
            QListWidget::item {{ padding: 6px 12px; }}
            QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
            QListWidget::item:hover {{ background: {BG}; }}
        """)
        self.list.itemActivated.connect(self._on_activated)

        lay.addWidget(self.input)
        lay.addWidget(self.list)

    # ---- public ----
    def set_commands(self, commands: list[Command]):
        self._commands = list(commands)
        self._refilter("")

    def open(self):
        self.input.clear()
        self._refilter("")
        self.show()
        self.input.setFocus()
        parent = self.parent()
        if parent is not None:
            center = parent.frameGeometry().center()
            self.move(center.x() - self.width() // 2, parent.geometry().top() + 80)

    # ---- filtering ----
    def _refilter(self, text: str):
        text = text.strip().lower()
        self.list.clear()
        if not text:
            items = self._commands[:50]
        else:
            items = [c for c in self._commands if _fuzzy_match(text, c.label.lower() + " " + c.hint.lower())]
            items.sort(key=lambda c: (_score(text, c.label.lower()), c.label))
            items = items[:50]

        for c in items:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, c)
            self.list.addItem(item)
            w = _CommandRow(c)
            item.setSizeHint(w.sizeHint())
            self.list.setItemWidget(item, w)

        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _activate_current(self):
        item = self.list.currentItem()
        if item:
            self._on_activated(item)

    def _on_activated(self, item: QListWidgetItem):
        cmd: Command = item.data(Qt.UserRole)
        self.accept()
        cmd.action()

    # ---- keyboard ----
    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key_Down,):
            self.list.setCurrentRow(min(self.list.currentRow() + 1, self.list.count() - 1))
            return
        if e.key() in (Qt.Key_Up,):
            self.list.setCurrentRow(max(self.list.currentRow() - 1, 0))
            return
        if e.key() == Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(e)


class _CommandRow(QFrame):
    def __init__(self, cmd: Command):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        label = QLabel(cmd.label)
        label.setFont(ui_font(12, 500))
        label.setStyleSheet(f"color: {TEXT};")
        hint = QLabel(cmd.hint)
        hint.setFont(mono_font(11))
        hint.setStyleSheet(f"color: {TEXT_SECONDARY};")
        lay.addWidget(label)
        lay.addStretch(1)
        lay.addWidget(hint)


def _fuzzy_match(needle: str, haystack: str) -> bool:
    i = 0
    for ch in haystack:
        if i < len(needle) and needle[i] == ch:
            i += 1
    return i == len(needle)


def _score(needle: str, label: str) -> int:
    # prefer prefix matches, then contains, then general fuzzy
    if label.startswith(needle):
        return 0
    if needle in label:
        return 1
    return 2
