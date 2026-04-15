"""Portfolio workspace — NAV bar, holdings/trades tabs, concentration panel.

Replaces the original PortfolioService-backed workspace with a PaperService
adapter that shares live state with the TUI.

Layout
------
PaneHeader "Portfolio"  (subtitle = live NAV inline text)
── NAV bar (36 px, 10 KPI inline pairs) ────────────────────────────────────────
── QSplitter(Horizontal) ───────────────────────────────────────────────────────
   LEFT  (60%): QTabWidget
     Tab "Holdings" — 12-column positions table
     Tab "Trades"   — 9-column trade history table
   RIGHT (40%): _ConcentrationPanel  (positions + sector bars via QPainter)
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSize,
    QTimer,
)
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QFontMetrics,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.theme import (
    BG,
    PANE,
    ELEVATED,
    BORDER,
    TEXT,
    TEXT_SECONDARY,
    TEXT_MUTED,
    GAIN,
    LOSS,
    ACCENT,
    WARN,
    mono_font,
    ui_font,
)
from apps.desktop.utils import fmt_number, fmt_signed, fmt_signed_pct, fmt_volume
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.pane_header import PaneHeader
from apps.desktop.services.paper_service import PaperService
from apps.desktop.services.paper_types import (
    PaperPosition,
    NavSummary,
    ConcentrationRow,
    Trade,
)

# ── Column definitions ────────────────────────────────────────────────────────

_HOLDINGS_COLS: list[tuple[str, str, int, str]] = [
    ("symbol",         "SYMBOL",  88,  "left"),
    ("qty",            "QTY",     70,  "right"),
    ("avg_price",      "AVG",     80,  "right"),
    ("last_price",     "LAST",    80,  "right"),
    ("day_pct",        "DAY%",    68,  "right"),
    ("market_value",   "MKT VAL", 100, "right"),
    ("unrealized_pnl", "P&L",     96,  "right"),
    ("pct_return",     "P&L%",    72,  "right"),
    ("days_held",      "DAYS",    52,  "right"),
    ("signal_type",    "SIGNAL",  96,  "left"),
    ("sector",         "SECTOR",  108, "left"),
    ("weight_pct",     "WEIGHT",  64,  "right"),
]

_TRADES_COLS: list[tuple[str, str, int, str]] = [
    ("date",    "DATE",   110, "left"),
    ("action",  "ACTION",  62, "left"),
    ("symbol",  "SYMBOL",  88, "left"),
    ("shares",  "QTY",     70, "right"),
    ("price",   "PRICE",   84, "right"),
    ("fees",    "FEES",    76, "right"),
    ("pnl",     "P&L",     90, "right"),
    ("pnl_pct", "P&L%",    72, "right"),
    ("reason",  "REASON",   0, "left"),   # 0 = stretch last
]

# Weight thresholds for colour coding
_WEIGHT_WARN  = 25.0
_WEIGHT_CRIT  = 35.0

# Minimum row height in tables
_ROW_H = 26


# ── Holdings table model ───────────────────────────────────────────────────────

class _HoldingsModel(QAbstractTableModel):
    """Qt model for the live positions table."""

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[PaperPosition] = []
        self._font      = mono_font(12)
        self._font_bold = mono_font(12, 600)

    # ── QAbstractTableModel interface ─────────────────────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(_HOLDINGS_COLS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return _HOLDINGS_COLS[section][1]
            if role == Qt.TextAlignmentRole:
                align = _HOLDINGS_COLS[section][3]
                h = Qt.AlignLeft if align == "left" else Qt.AlignRight
                return int(h | Qt.AlignVCenter)
        return None

    def data(self, idx: QModelIndex, role: int = Qt.DisplayRole):
        if not idx.isValid():
            return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows):
            return None
        p = self._rows[r]
        key = _HOLDINGS_COLS[c][0]

        if role == Qt.DisplayRole:
            return self._display(p, key)

        if role == Qt.TextAlignmentRole:
            align = _HOLDINGS_COLS[c][3]
            h = Qt.AlignLeft if align == "left" else Qt.AlignRight
            return int(h | Qt.AlignVCenter)

        if role == Qt.ForegroundRole:
            return QColor(self._fg(p, key))

        if role == Qt.FontRole:
            return self._font_bold if key == "symbol" else self._font

        if role == Qt.UserRole:
            return p

        return None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _display(self, p: PaperPosition, key: str) -> str:
        if key == "symbol":
            return p.symbol
        if key == "qty":
            return f"{p.qty:,.0f}"
        if key == "avg_price":
            return fmt_number(p.avg_price)
        if key == "last_price":
            return fmt_number(p.last_price)
        if key == "day_pct":
            return fmt_signed_pct(p.day_pct)
        if key == "market_value":
            return fmt_number(p.market_value)
        if key == "unrealized_pnl":
            return fmt_signed(p.unrealized_pnl)
        if key == "pct_return":
            return fmt_signed_pct(p.pct_return)
        if key == "days_held":
            return str(p.days_held)
        if key == "signal_type":
            return p.signal_type
        if key == "sector":
            return p.sector
        if key == "weight_pct":
            return f"{p.weight_pct:.1f}%"
        return ""

    def _fg(self, p: PaperPosition, key: str) -> str:
        if key == "symbol":
            return TEXT
        if key == "day_pct":
            return GAIN if p.day_pct >= 0 else LOSS
        if key == "unrealized_pnl":
            return GAIN if p.unrealized_pnl >= 0 else LOSS
        if key == "pct_return":
            return GAIN if p.pct_return >= 0 else LOSS
        if key == "weight_pct":
            if p.weight_pct >= _WEIGHT_CRIT:
                return LOSS
            if p.weight_pct >= _WEIGHT_WARN:
                return WARN
            return TEXT
        return TEXT

    # ── mutation ──────────────────────────────────────────────────────────────

    def set_rows(self, rows: list[PaperPosition]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, r: int) -> Optional[PaperPosition]:
        if 0 <= r < len(self._rows):
            return self._rows[r]
        return None


# ── Trade history table model ─────────────────────────────────────────────────

class _TradesModel(QAbstractTableModel):
    """Qt model for the trade history table."""

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[Trade] = []
        self._font      = mono_font(12)
        self._font_bold = mono_font(12, 600)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(_TRADES_COLS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return _TRADES_COLS[section][1]
            if role == Qt.TextAlignmentRole:
                align = _TRADES_COLS[section][3]
                h = Qt.AlignLeft if align == "left" else Qt.AlignRight
                return int(h | Qt.AlignVCenter)
        return None

    def data(self, idx: QModelIndex, role: int = Qt.DisplayRole):
        if not idx.isValid():
            return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows):
            return None
        t = self._rows[r]
        key = _TRADES_COLS[c][0]

        if role == Qt.DisplayRole:
            return self._display(t, key)

        if role == Qt.TextAlignmentRole:
            align = _TRADES_COLS[c][3]
            h = Qt.AlignLeft if align == "left" else Qt.AlignRight
            return int(h | Qt.AlignVCenter)

        if role == Qt.ForegroundRole:
            return QColor(self._fg(t, key))

        if role == Qt.FontRole:
            return self._font_bold if key == "symbol" else self._font

        return None

    def _display(self, t: Trade, key: str) -> str:
        if key == "date":
            return t.date[:10] if t.date else ""
        if key == "action":
            return t.action
        if key == "symbol":
            return t.symbol
        if key == "shares":
            return f"{t.shares:,.0f}"
        if key == "price":
            return fmt_number(t.price)
        if key == "fees":
            return fmt_number(t.fees)
        if key == "pnl":
            return fmt_signed(t.pnl) if t.action.upper() == "SELL" else "—"
        if key == "pnl_pct":
            return fmt_signed_pct(t.pnl_pct) if t.action.upper() == "SELL" else "—"
        if key == "reason":
            return t.reason
        return ""

    def _fg(self, t: Trade, key: str) -> str:
        if key == "action":
            return GAIN if t.action.upper() == "BUY" else LOSS
        if key == "pnl":
            return GAIN if t.pnl >= 0 else LOSS
        if key == "pnl_pct":
            return GAIN if t.pnl_pct >= 0 else LOSS
        return TEXT

    def set_rows(self, rows: list[Trade]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()


# ── NAV bar ────────────────────────────────────────────────────────────────────

class _NavBar(QFrame):
    """Single-row 36 px bar containing 10 inline label/value KPI pairs."""

    # Ordered list of (attr_on_NavSummary_or_None, display_label)
    _FIELDS = [
        ("nav",          "NAV"),
        ("cash",         "Cash"),
        ("invested",     "Invested"),
        ("day_pnl",      "Day"),
        ("day_pct",      "Day%"),
        ("total_return", "Net%"),
        ("nepse_return", "NEPSE%"),
        ("alpha",        "Alpha"),
        ("max_dd",       "MaxDD"),
        ("n_positions",  "Pos"),
    ]

    # Fields that should be sign-coloured
    _SIGNED = {"day_pnl", "day_pct", "total_return", "alpha"}
    # Fields formatted as percentage numbers
    _PCT    = {"day_pct", "total_return", "nepse_return", "alpha", "max_dd"}
    # Fields formatted as integer counts
    _INT    = {"n_positions"}
    # Fields formatted as currency
    _CURR   = {"nav", "cash", "invested", "day_pnl"}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(
            f"_NavBar {{ background: {ELEVATED}; border-bottom: 1px solid {BORDER}; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(0)

        self._value_labels: dict[str, QLabel] = {}

        for i, (key, lbl) in enumerate(self._FIELDS):
            # Separator between pairs (not before first)
            if i > 0:
                sep = QLabel("·")
                sep.setFont(ui_font(11))
                sep.setStyleSheet(f"color: {BORDER}; padding: 0 8px;")
                lay.addWidget(sep, 0, Qt.AlignVCenter)

            # Label
            l_lbl = QLabel(lbl.upper())
            l_lbl.setFont(ui_font(10, 500))
            l_lbl.setStyleSheet(
                f"color: {TEXT_MUTED}; letter-spacing: 0.6px; padding-right: 4px;"
            )
            lay.addWidget(l_lbl, 0, Qt.AlignVCenter)

            # Value
            v_lbl = QLabel("—")
            v_lbl.setFont(mono_font(12, 500))
            v_lbl.setStyleSheet(f"color: {TEXT};")
            v_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lay.addWidget(v_lbl, 0, Qt.AlignVCenter)
            self._value_labels[key] = v_lbl

        lay.addStretch(1)

    def update_summary(self, s: NavSummary) -> None:
        """Push a fresh NavSummary into all label widgets."""
        for key, _ in self._FIELDS:
            lbl = self._value_labels.get(key)
            if lbl is None:
                continue
            val = getattr(s, key, None)
            if val is None:
                lbl.setText("—")
                lbl.setStyleSheet(f"color: {TEXT_MUTED};")
                continue

            # Format
            if key in self._INT:
                text = str(int(val))
            elif key in self._PCT:
                text = fmt_signed_pct(float(val)) if key in self._SIGNED else f"{float(val):.2f}%"
            elif key in self._CURR:
                text = fmt_signed(float(val)) if key in self._SIGNED else fmt_number(float(val))
            else:
                text = str(val)

            # Colour
            if key in self._SIGNED:
                colour = GAIN if float(val) >= 0 else LOSS
            elif key == "max_dd":
                colour = LOSS if float(val) < -5.0 else WARN if float(val) < 0 else GAIN
            else:
                colour = TEXT

            lbl.setText(text)
            lbl.setStyleSheet(f"color: {colour};")

    def clear(self) -> None:
        """Reset all values to em-dash placeholders."""
        for lbl in self._value_labels.values():
            lbl.setText("—")
            lbl.setStyleSheet(f"color: {TEXT_MUTED};")


# ── Concentration panel ────────────────────────────────────────────────────────

class _ConcentrationPanel(QWidget):
    """Right panel: horizontal bar charts for top-5 positions and sector weights.

    Entirely rendered via QPainter — no child widgets, no QTableView.
    """

    _BAR_H        = 18    # px height of each bar
    _BAR_SPACING  = 6     # px between bars
    _SECTION_GAP  = 20    # px gap between position and sector sections
    _LABEL_W      = 72    # px width reserved for text labels on the left
    _PCT_W        = 42    # px width reserved for percentage text on the right
    _MARGIN_H     = 12    # horizontal outer margin
    _MARGIN_V     = 12    # vertical outer margin
    _HEADER_H     = 18    # px for section header text
    _SEP_H        = 1     # px separator line height
    _SEP_MARGIN   = 8     # px above/below separator

    _BAR_COLOR_POS         = QColor(ACCENT)
    _BAR_COLOR_SECTOR      = QColor("#4D9FFF").darker(110)
    _BAR_COLOR_WARN        = QColor(WARN)
    _BAR_COLOR_CRIT        = QColor(LOSS)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: list[ConcentrationRow] = []
        self.setMinimumWidth(180)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.setStyleSheet(f"background: {PANE};")

    def set_rows(self, rows: list[ConcentrationRow]) -> None:
        self._rows = list(rows)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(260, 400)

    # ── paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        # Background
        p.fillRect(self.rect(), QColor(PANE))

        if not self._rows:
            p.setPen(QColor(TEXT_MUTED))
            p.setFont(ui_font(11))
            p.drawText(self.rect(), Qt.AlignCenter, "No positions")
            return

        pos_rows    = [r for r in self._rows if r.row_type == "POSITION"]
        sector_rows = [r for r in self._rows if r.row_type == "SECTOR"]

        w = self.width()
        x = self._MARGIN_H
        bar_w = w - self._MARGIN_H * 2 - self._LABEL_W - self._PCT_W - 8
        if bar_w < 20:
            bar_w = 20

        y = self._MARGIN_V

        # ── positions section ──────────────────────────────────────────────
        y = self._draw_section_header(p, x, y, "POSITIONS", w)
        for row in pos_rows:
            y = self._draw_bar(p, x, y, row, bar_w, is_sector=False)
            y += self._BAR_SPACING

        if pos_rows and sector_rows:
            y += self._SECTION_GAP // 2
            # Separator
            p.setPen(QPen(QColor(BORDER), 1))
            p.drawLine(x, y, w - self._MARGIN_H, y)
            y += self._SEP_H + self._SECTION_GAP // 2

        # ── sector section ─────────────────────────────────────────────────
        if sector_rows:
            y = self._draw_section_header(p, x, y, "SECTORS", w)
            for row in sector_rows:
                y = self._draw_bar(p, x, y, row, bar_w, is_sector=True)
                y += self._BAR_SPACING

        p.end()

    def _draw_section_header(
        self,
        p: QPainter,
        x: int,
        y: int,
        title: str,
        total_w: int,
    ) -> int:
        """Draw a section title. Returns the new Y cursor after the header."""
        p.setFont(ui_font(10, 600))
        p.setPen(QColor(TEXT_SECONDARY))
        p.drawText(
            x, y,
            total_w - self._MARGIN_H * 2,
            self._HEADER_H,
            Qt.AlignLeft | Qt.AlignVCenter,
            title,
        )
        return y + self._HEADER_H + 4

    def _draw_bar(
        self,
        p: QPainter,
        x: int,
        y: int,
        row: ConcentrationRow,
        bar_w: int,
        is_sector: bool,
    ) -> int:
        """Draw a single labelled bar row. Returns the new Y cursor."""
        # Determine bar colour
        if is_sector:
            bar_color = self._BAR_COLOR_SECTOR
        elif row.weight_pct >= _WEIGHT_CRIT:
            bar_color = self._BAR_COLOR_CRIT
        elif row.weight_pct >= _WEIGHT_WARN:
            bar_color = self._BAR_COLOR_WARN
        else:
            bar_color = self._BAR_COLOR_POS

        # Label (left)
        label = row.name[:10]  # truncate long names
        p.setFont(mono_font(11))
        p.setPen(QColor(TEXT_SECONDARY))
        p.drawText(
            x, y,
            self._LABEL_W,
            self._BAR_H,
            Qt.AlignLeft | Qt.AlignVCenter,
            label,
        )

        # Bar background
        bar_x = x + self._LABEL_W + 4
        p.setPen(QPen(Qt.NoPen))
        p.setBrush(QBrush(QColor(ELEVATED)))
        p.drawRect(bar_x, y + 3, bar_w, self._BAR_H - 6)

        # Bar fill
        fill_w = max(2, int(min(row.weight_pct, 100.0) / 100.0 * bar_w))
        p.setBrush(QBrush(bar_color))
        p.drawRect(bar_x, y + 3, fill_w, self._BAR_H - 6)

        # Percentage (right)
        pct_text = f"{row.weight_pct:.1f}%"
        pct_color = (
            LOSS if row.weight_pct >= _WEIGHT_CRIT
            else WARN if row.weight_pct >= _WEIGHT_WARN
            else TEXT_SECONDARY
        )
        p.setFont(mono_font(11))
        p.setPen(QColor(pct_color))
        p.drawText(
            bar_x + bar_w + 4, y,
            self._PCT_W,
            self._BAR_H,
            Qt.AlignLeft | Qt.AlignVCenter,
            pct_text,
        )

        return y + self._BAR_H


# ── Holdings tab widget ────────────────────────────────────────────────────────

def _make_table(model: QAbstractTableModel, cols: list[tuple], stretch_last: bool = False) -> QTableView:
    """Build a styled QTableView for the given model and column spec."""
    tv = QTableView()
    tv.setModel(model)
    tv.setShowGrid(False)
    tv.verticalHeader().setVisible(False)
    tv.verticalHeader().setDefaultSectionSize(_ROW_H)
    tv.horizontalHeader().setHighlightSections(False)
    tv.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    if stretch_last:
        tv.horizontalHeader().setStretchLastSection(True)
    tv.setSelectionBehavior(QTableView.SelectRows)
    tv.setEditTriggers(QTableView.NoEditTriggers)
    tv.setAlternatingRowColors(False)
    tv.setStyleSheet(
        f"""
        QTableView {{
            background: {PANE};
            color: {TEXT};
            gridline-color: {BORDER};
            selection-background-color: {ELEVATED};
            selection-color: {TEXT};
            border: none;
        }}
        QTableView::item {{
            padding: 0 6px;
            border: none;
        }}
        QHeaderView::section {{
            background: {BG};
            color: {TEXT_SECONDARY};
            border: none;
            border-bottom: 1px solid {BORDER};
            padding: 0 6px;
            font-size: 11px;
            letter-spacing: 0.6px;
        }}
        """
    )
    for i, (_k, _h, w, _a) in enumerate(cols):
        if w > 0:
            tv.setColumnWidth(i, w)
    return tv


# ── Empty placeholder ──────────────────────────────────────────────────────────

def _make_empty_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setFont(ui_font(12))
    lbl.setStyleSheet(
        f"color: {TEXT_MUTED}; padding: 40px; background: {PANE};"
    )
    lbl.setWordWrap(True)
    return lbl


# ── Tab styling helper ─────────────────────────────────────────────────────────

_TAB_SS = f"""
QTabWidget::pane {{
    border: none;
    background: {PANE};
}}
QTabBar::tab {{
    background: {BG};
    color: {TEXT_SECONDARY};
    padding: 5px 14px;
    font-size: 11px;
    border: none;
    border-right: 1px solid {BORDER};
}}
QTabBar::tab:selected {{
    background: {PANE};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{
    background: {ELEVATED};
    color: {TEXT};
}}
"""


# ── Main Portfolio workspace ───────────────────────────────────────────────────

class Portfolio(QWidget):
    """Paper trading portfolio workspace.

    Replaces the old PortfolioService-backed workspace with PaperService so the
    desktop and TUI share the same live state files.
    """

    _REFRESH_MS = 30_000   # 30 s auto-refresh

    def __init__(
        self,
        service: Optional[PaperService] = None,
        link: Optional[LinkGroup] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._service = service or PaperService()
        self._link    = link or LinkGroup()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────────
        self.header = PaneHeader(
            "Portfolio",
            link_color=self._link.color,
            subtitle="",
        )
        root.addWidget(self.header)

        # ── NAV bar ─────────────────────────────────────────────────────────
        self._nav_bar = _NavBar()
        root.addWidget(self._nav_bar)

        # ── Splitter ─────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {BORDER}; }}"
        )
        root.addWidget(splitter, 1)

        # ── Left: tab widget ─────────────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(_TAB_SS)
        self._tabs.setDocumentMode(True)
        left_lay.addWidget(self._tabs)

        # Holdings tab
        self._holdings_container = QWidget()
        hc_lay = QVBoxLayout(self._holdings_container)
        hc_lay.setContentsMargins(0, 0, 0, 0)
        hc_lay.setSpacing(0)

        self._holdings_model = _HoldingsModel()
        self._holdings_table = _make_table(
            self._holdings_model, _HOLDINGS_COLS, stretch_last=False
        )
        self._holdings_table.clicked.connect(self._on_holding_clicked)
        hc_lay.addWidget(self._holdings_table)

        self._holdings_empty = _make_empty_label(
            "No open positions.\n\nSignals will appear here once the strategy\n"
            "allocates capital during a trading session."
        )
        hc_lay.addWidget(self._holdings_empty)
        self._holdings_empty.hide()

        self._tabs.addTab(self._holdings_container, "Holdings")

        # Trades tab
        self._trades_container = QWidget()
        tc_lay = QVBoxLayout(self._trades_container)
        tc_lay.setContentsMargins(0, 0, 0, 0)
        tc_lay.setSpacing(0)

        self._trades_model = _TradesModel()
        self._trades_table = _make_table(
            self._trades_model, _TRADES_COLS, stretch_last=True
        )
        tc_lay.addWidget(self._trades_table)

        self._trades_empty = _make_empty_label(
            "No trade history yet.\n\nCompleted buys and sells will appear here."
        )
        tc_lay.addWidget(self._trades_empty)
        self._trades_empty.hide()

        self._tabs.addTab(self._trades_container, "Trades")

        splitter.addWidget(left)

        # ── Right: concentration panel ────────────────────────────────────────
        right = QWidget()
        right.setStyleSheet(
            f"QWidget {{ background: {PANE}; border-left: 1px solid {BORDER}; }}"
        )
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        # Right panel header
        conc_header = QFrame()
        conc_header.setFixedHeight(26)
        conc_header.setStyleSheet(
            f"QFrame {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}"
        )
        ch_lay = QHBoxLayout(conc_header)
        ch_lay.setContentsMargins(10, 0, 10, 0)
        lbl_conc = QLabel("CONCENTRATION")
        lbl_conc.setFont(ui_font(11, 600))
        lbl_conc.setStyleSheet(
            f"color: {TEXT_SECONDARY}; letter-spacing: 1.2px; background: transparent;"
        )
        ch_lay.addWidget(lbl_conc)
        ch_lay.addStretch(1)
        right_lay.addWidget(conc_header)

        self._conc_panel = _ConcentrationPanel()
        right_lay.addWidget(self._conc_panel, 1)

        splitter.addWidget(right)

        # 60 / 40 initial split
        splitter.setSizes([600, 400])

        # ── Timer ────────────────────────────────────────────────────────────
        QTimer.singleShot(60, self._refresh)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(self._REFRESH_MS)

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Reload all data from PaperService and push to all sub-widgets."""
        self._refresh_holdings()
        self._refresh_trades()
        self._refresh_nav()
        self._refresh_concentration()

    def _refresh_holdings(self) -> None:
        """Reload positions and update holdings table."""
        try:
            positions = self._service.positions()
        except Exception:
            positions = []

        self._holdings_model.set_rows(positions)

        if not positions:
            self._holdings_table.hide()
            self._holdings_empty.show()
        else:
            self._holdings_empty.hide()
            self._holdings_table.show()

    def _refresh_trades(self) -> None:
        """Reload trade history and update trades table."""
        try:
            trades = self._service.trade_history()
        except Exception:
            trades = []

        self._trades_model.set_rows(trades)

        if not trades:
            self._trades_table.hide()
            self._trades_empty.show()
        else:
            self._trades_empty.hide()
            self._trades_table.show()

    def _refresh_nav(self) -> None:
        """Reload NAV summary and push to the NAV bar and header subtitle."""
        try:
            summary = self._service.nav_summary()
        except Exception:
            self._nav_bar.clear()
            self.header.set_subtitle("—")
            return

        self._nav_bar.update_summary(summary)

        # Inline subtitle for the header
        nav_text = fmt_number(summary.nav)
        pos_text = f"{summary.n_positions} pos"
        ret_text = fmt_signed_pct(summary.total_return)
        dd_text  = f"DD {summary.max_dd:.1f}%"
        self.header.set_subtitle(
            f"{nav_text}  ·  {pos_text}  ·  {ret_text}  ·  {dd_text}"
        )

    def _refresh_concentration(self) -> None:
        """Reload concentration data and push to the panel."""
        try:
            rows = self._service.concentration()
        except Exception:
            rows = []
        self._conc_panel.set_rows(rows)

    # ── Interaction ───────────────────────────────────────────────────────────

    def _on_holding_clicked(self, idx: QModelIndex) -> None:
        """Propagate symbol selection to the link group."""
        if not idx.isValid():
            return
        pos = self._holdings_model.row_at(idx.row())
        if pos is not None:
            self._link.set_symbol(pos.symbol)

    # ── Public API ────────────────────────────────────────────────────────────

    def force_refresh(self) -> None:
        """Trigger an immediate data refresh (callable from parent workspaces)."""
        self._refresh()

    def set_service(self, service: PaperService) -> None:
        """Swap the underlying service (e.g. on account switch)."""
        self._service = service
        self._refresh()
