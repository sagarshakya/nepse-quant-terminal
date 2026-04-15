"""Dark terminal theme. Colors exactly per Build.md §9.

No gradients. No shadows (except menus). Color carries meaning, never decoration.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtCore import Qt


# ---- palette ----------------------------------------------------------------
BG = "#0C0F12"
PANE = "#14181D"
ELEVATED = "#1A1F26"
BORDER = "#242A32"
BORDER_STRONG = "#2F363F"

TEXT = "#E6E8EB"
TEXT_SECONDARY = "#8A93A0"
TEXT_MUTED = "#5B626C"

ACCENT = "#4D9FFF"
ACCENT_SOFT = "#1E3A5F"

GAIN = "#2AC27D"
GAIN_HI = "#5EE39C"
LOSS = "#E5484D"
LOSS_HI = "#FF6B6E"
WARN = "#E0A23A"
CRIT = "#CD4246"

LINK_BLUE = "#4D9FFF"
LINK_RED = "#E5484D"
LINK_GREEN = "#2AC27D"
LINK_YELLOW = "#E0A23A"


# ---- fonts ------------------------------------------------------------------
def _weight(w: int) -> QFont.Weight:
    """Map numeric CSS weight (100..900) to QFont.Weight enum."""
    table = [
        (150, QFont.Weight.Thin),
        (250, QFont.Weight.ExtraLight),
        (350, QFont.Weight.Light),
        (450, QFont.Weight.Normal),
        (525, QFont.Weight.Medium),
        (625, QFont.Weight.DemiBold),
        (750, QFont.Weight.Bold),
        (850, QFont.Weight.ExtraBold),
        (1000, QFont.Weight.Black),
    ]
    for thresh, qw in table:
        if w <= thresh:
            return qw
    return QFont.Weight.Normal


def ui_font(size: int = 12, weight: int = 400) -> QFont:
    f = QFont("Inter")
    if not f.exactMatch():
        f = QFont("SF Pro Text")
    if not f.exactMatch():
        f = QFont("Helvetica Neue")
    f.setPixelSize(size)
    f.setWeight(_weight(weight))
    return f


def mono_font(size: int = 12, weight: int = 400) -> QFont:
    for family in ("JetBrains Mono", "IBM Plex Mono", "SF Mono", "Menlo", "Monaco"):
        f = QFont(family)
        if f.exactMatch() or family in ("Menlo", "Monaco"):
            f.setPixelSize(size)
            f.setWeight(_weight(weight))
            f.setStyleHint(QFont.Monospace)
            return f
    f = QFont("Courier New")
    f.setPixelSize(size)
    f.setWeight(_weight(weight))
    f.setStyleHint(QFont.Monospace)
    return f


# ---- stylesheet -------------------------------------------------------------
def stylesheet() -> str:
    return f"""
    /* ------------------ globals ------------------ */
    QWidget {{
        background-color: {BG};
        color: {TEXT};
        font-family: "Inter", "SF Pro Text", "Helvetica Neue", sans-serif;
        font-size: 12px;
    }}

    QToolTip {{
        background-color: {ELEVATED};
        color: {TEXT};
        border: 1px solid {BORDER};
        padding: 4px 6px;
    }}

    /* ------------------ main window ------------------ */
    QMainWindow::separator {{
        background: {BORDER};
        width: 1px;
        height: 1px;
    }}

    QStatusBar {{
        background: {PANE};
        color: {TEXT_SECONDARY};
        border-top: 1px solid {BORDER};
    }}

    /* ------------------ dock widgets ------------------ */
    QDockWidget {{
        color: {TEXT_SECONDARY};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background: {PANE};
        border-bottom: 1px solid {BORDER};
        padding: 4px 8px;
        text-align: left;
        font-size: 11px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    QDockWidget > QWidget {{
        background: {PANE};
    }}

    /* ------------------ splitters ------------------ */
    QSplitter::handle {{ background: {BORDER}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical   {{ height: 1px; }}

    /* ------------------ tables ------------------ */
    QTableView, QTableWidget {{
        background-color: {PANE};
        alternate-background-color: {PANE};
        gridline-color: transparent;
        selection-background-color: {ACCENT_SOFT};
        selection-color: {TEXT};
        border: none;
        outline: 0;
    }}
    QTableView::item, QTableWidget::item {{
        padding: 2px 6px;
        border: 0;
    }}
    QTableView::item:hover {{ background-color: {ELEVATED}; }}
    QTableView::item:selected {{ background-color: {ACCENT_SOFT}; color: {TEXT}; }}

    QHeaderView::section {{
        background-color: {PANE};
        color: {TEXT_SECONDARY};
        border: 0;
        border-bottom: 1px solid {BORDER};
        padding: 4px 6px;
        font-weight: 500;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    QHeaderView::section:hover {{ color: {TEXT}; }}

    /* ------------------ scrollbars ------------------ */
    QScrollBar:vertical {{
        background: {PANE};
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_STRONG};
        min-height: 24px;
        border-radius: 2px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {TEXT_MUTED}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

    QScrollBar:horizontal {{
        background: {PANE};
        height: 10px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_STRONG};
        min-width: 24px;
        border-radius: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {TEXT_MUTED}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

    /* ------------------ inputs ------------------ */
    QLineEdit {{
        background: {BG};
        color: {TEXT};
        border: 1px solid {BORDER};
        padding: 4px 6px;
        selection-background-color: {ACCENT_SOFT};
    }}
    QLineEdit:focus {{ border: 1px solid {ACCENT}; }}

    QPushButton {{
        background: {PANE};
        color: {TEXT};
        border: 1px solid {BORDER};
        padding: 4px 10px;
    }}
    QPushButton:hover {{ border-color: {TEXT_MUTED}; }}
    QPushButton:pressed {{ background: {ELEVATED}; }}
    QPushButton:default {{ border-color: {ACCENT}; color: {ACCENT}; }}

    /* ------------------ menus ------------------ */
    QMenu {{
        background: {ELEVATED};
        color: {TEXT};
        border: 1px solid {BORDER};
        padding: 4px 0;
    }}
    QMenu::item {{ padding: 4px 16px; }}
    QMenu::item:selected {{ background: {ACCENT_SOFT}; }}

    QMenuBar {{ background: {PANE}; border-bottom: 1px solid {BORDER}; }}
    QMenuBar::item:selected {{ background: {ELEVATED}; }}

    /* ------------------ list views ------------------ */
    QListView, QListWidget {{
        background: {PANE};
        border: none;
        outline: 0;
    }}
    QListView::item, QListWidget::item {{ padding: 4px 8px; }}
    QListView::item:hover, QListWidget::item:hover {{ background: {ELEVATED}; }}
    QListView::item:selected, QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
    """


def apply_palette(app):
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(PANE))
    pal.setColor(QPalette.AlternateBase, QColor(ELEVATED))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(PANE))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT_SOFT))
    pal.setColor(QPalette.HighlightedText, QColor(TEXT))
    pal.setColor(QPalette.ToolTipBase, QColor(ELEVATED))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet())
