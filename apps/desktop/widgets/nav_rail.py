"""Left-side 48px navigation rail for workspace switching."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QToolButton, QVBoxLayout, QWidget,
)

from apps.desktop.theme import (
    PANE, BORDER, TEXT, TEXT_SECONDARY, ACCENT, ACCENT_SOFT, ELEVATED, ui_font,
)


class _RailButton(QToolButton):
    def __init__(self, label: str, glyph: str, tooltip: str):
        super().__init__()
        self.label = label
        self.setText(glyph)
        self.setToolTip(tooltip)
        self.setFont(ui_font(18, 500))
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setFixedSize(40, 40)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                color: {TEXT_SECONDARY};
                border: 1px solid transparent;
                padding: 0;
            }}
            QToolButton:hover {{
                color: {TEXT};
                background: {ELEVATED};
            }}
            QToolButton:checked {{
                color: {ACCENT};
                background: {ACCENT_SOFT};
                border-left: 2px solid {ACCENT};
            }}
        """)


class NavRail(QFrame):
    """Icon-only vertical nav. Workspace switching.

    Parameters
    ----------
    items:
        Optional list of ``(key, glyph, tooltip)`` tuples that define the
        workspace buttons.  When *None* (the historical default) a single
        "ticker" button is created for backward compatibility.
    parent:
        Optional parent widget.
    """

    workspace_selected = Signal(str)  # emits workspace key

    def __init__(
        self,
        items: Optional[list[tuple[str, str, str]]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setFixedWidth(48)
        self.setStyleSheet(
            f"NavRail {{ background: {PANE}; border-right: 1px solid {BORDER}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 8, 4, 8)
        lay.setSpacing(2)

        # Fall back to the original single-item list when no items provided
        if items is None:
            items = [
                ("ticker", "◉", "Price  (Ctrl+1)"),
            ]

        self._buttons: dict[str, _RailButton] = {}
        for key, glyph, tip in items:
            btn = _RailButton(key, glyph, tip)
            btn.clicked.connect(lambda checked=False, k=key: self._on_clicked(k))
            self._buttons[key] = btn
            lay.addWidget(btn)

        lay.addStretch(1)

        # Settings at bottom (never part of the workspace auto-exclusive group)
        settings = _RailButton("settings", "⚙", "Settings")
        settings.setAutoExclusive(False)
        lay.addWidget(settings)

        # Default selection — first workspace button
        if self._buttons:
            first_key = next(iter(self._buttons))
            self._buttons[first_key].setChecked(True)

    def _on_clicked(self, key: str) -> None:
        self.workspace_selected.emit(key)

    def set_active(self, key: str) -> None:
        if key in self._buttons:
            self._buttons[key].setChecked(True)
