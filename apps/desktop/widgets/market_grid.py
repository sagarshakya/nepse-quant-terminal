"""Virtualized market grid with sparklines, colored % cells, live filter."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QTimer, QSize,
    QPointF, QRectF,
)
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont,
)
from PySide6.QtWidgets import (
    QFrame, QHeaderView, QLineEdit, QTableView, QVBoxLayout, QWidget, QLabel,
    QHBoxLayout, QStyledItemDelegate, QStyleOptionViewItem,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, TEXT, TEXT_SECONDARY, TEXT_MUTED,
    GAIN, LOSS, ACCENT, ACCENT_SOFT, mono_font, ui_font,
)
from apps.desktop.utils import fmt_number, fmt_signed, fmt_signed_pct, fmt_volume
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.pane_header import PaneHeader
from backend.core.services import MarketService
from backend.core.types import Quote


# (key, header, width, align)
COLUMNS = [
    ("symbol",  "SYMBOL",    88, "left"),
    ("last",    "LAST",      84, "right"),
    ("pct",     "CHG%",      78, "right"),
    ("change",  "CHG",       84, "right"),
    ("spark",   "30D",      100, "center"),
    ("open",    "OPEN",      80, "right"),
    ("high",    "HIGH",      80, "right"),
    ("low",     "LOW",       80, "right"),
    ("volume",  "VOL",       90, "right"),
    ("range",  "RANGE",     100, "right"),
]
COL_INDEX = {k: i for i, (k, *_r) in enumerate(COLUMNS)}


class _MarketModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list[Quote] = []
        self._sparks: dict[str, np.ndarray] = {}
        self._font_mono = mono_font(12, 400)
        self._font_bold = mono_font(12, 500)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return COLUMNS[section][1]
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            align = COLUMNS[section][3]
            if align == "left":
                return int(Qt.AlignLeft | Qt.AlignVCenter)
            if align == "center":
                return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def data(self, idx: QModelIndex, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows):
            return None
        q = self._rows[r]
        key = COLUMNS[c][0]

        if role == Qt.DisplayRole:
            return self._format(q, key)
        if role == Qt.TextAlignmentRole:
            align = COLUMNS[c][3]
            if align == "left":
                return int(Qt.AlignLeft | Qt.AlignVCenter)
            if align == "center":
                return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            if key in ("change", "pct"):
                if q.change > 0: return QColor(GAIN)
                if q.change < 0: return QColor(LOSS)
                return QColor(TEXT_MUTED)
            if key == "symbol":
                return QColor(TEXT)
            if key in ("volume", "range", "prev"):
                return QColor(TEXT_SECONDARY)
            return QColor(TEXT)
        if role == Qt.FontRole:
            return self._font_bold if key in ("symbol", "last", "pct") else self._font_mono
        if role == Qt.UserRole:
            return self._raw(q, key)
        if role == Qt.UserRole + 1 and key == "spark":
            return self._sparks.get(q.symbol)
        if role == Qt.UserRole + 2:
            return q  # full quote for delegates
        return None

    # formatting
    def _format(self, q: Quote, key: str) -> str:
        if key == "symbol":  return q.symbol
        if key == "last":    return fmt_number(q.last)
        if key == "change":  return fmt_signed(q.change)
        if key == "pct":     return fmt_signed_pct(q.change_pct)
        if key == "open":    return fmt_number(q.open)
        if key == "high":    return fmt_number(q.high)
        if key == "low":     return fmt_number(q.low)
        if key == "volume":  return fmt_volume(q.volume)
        if key == "spark":   return ""
        if key == "range":
            rng = q.high - q.low
            return fmt_number(rng)
        return ""

    def _raw(self, q: Quote, key: str):
        if key == "symbol":  return q.symbol
        if key == "last":    return q.last
        if key == "change":  return q.change
        if key == "pct":     return q.change_pct
        if key == "open":    return q.open
        if key == "high":    return q.high
        if key == "low":     return q.low
        if key == "volume":  return q.volume
        if key == "range":   return q.high - q.low
        if key == "spark":   return q.change_pct
        return ""

    # updates
    def set_rows(self, rows: list[Quote]):
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def set_sparks(self, sparks: dict[str, np.ndarray]):
        self._sparks = sparks
        # repaint spark column only
        top = self.index(0, COL_INDEX["spark"])
        bot = self.index(self.rowCount() - 1, COL_INDEX["spark"])
        if top.isValid() and bot.isValid():
            self.dataChanged.emit(top, bot, [Qt.UserRole + 1])

    def row_symbol(self, row: int) -> Optional[str]:
        if 0 <= row < len(self._rows):
            return self._rows[row].symbol
        return None


class _SparkDelegate(QStyledItemDelegate):
    """Draws a mini line-sparkline inside the cell."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        super().paint(painter, option, index)
        spark = index.data(Qt.UserRole + 1)
        if spark is None or len(spark) < 2:
            return
        rect = option.rect.adjusted(8, 4, -8, -4)
        arr = np.asarray(spark, dtype=np.float64)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi == lo:
            return
        w = rect.width(); h = rect.height()
        xs = np.linspace(rect.left(), rect.right(), arr.size)
        ys = rect.bottom() - (arr - lo) / (hi - lo) * h
        last_chg = arr[-1] - arr[0]
        color = QColor(GAIN) if last_chg >= 0 else QColor(LOSS)
        pen = QPen(color)
        pen.setWidthF(1.25)
        pen.setCosmetic(True)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(pen)
        for i in range(arr.size - 1):
            painter.drawLine(QPointF(xs[i], ys[i]), QPointF(xs[i + 1], ys[i + 1]))
        # end dot
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(Qt.NoPen))
        painter.drawEllipse(QPointF(xs[-1], ys[-1]), 2.0, 2.0)
        painter.restore()


class _SortProxy(QSortFilterProxyModel):
    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        a = self.sourceModel().data(left, Qt.UserRole)
        b = self.sourceModel().data(right, Qt.UserRole)
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        pattern = self.filterRegularExpression().pattern()
        if not pattern:
            return True
        sym_idx = self.sourceModel().index(source_row, COL_INDEX["symbol"], source_parent)
        sym = self.sourceModel().data(sym_idx, Qt.DisplayRole) or ""
        return pattern.upper() in sym.upper()


class MarketGrid(QFrame):
    """Live market grid, linked to a LinkGroup."""

    def __init__(self, market: MarketService, link: LinkGroup, parent=None):
        super().__init__(parent)
        self._market = market
        self._link = link
        self._spark_cache: dict[str, np.ndarray] = {}
        self._spark_loaded = False
        self._build()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(20_000)
        QTimer.singleShot(0, self.refresh)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Pane header
        self.header = PaneHeader("Market Grid", link_color=self._link.color, subtitle="")
        lay.addWidget(self.header)

        # Filter row
        filter_row = QFrame()
        filter_row.setStyleSheet(f"background: {PANE}; border-bottom: 1px solid {BORDER};")
        filter_row.setFixedHeight(30)
        fl = QHBoxLayout(filter_row)
        fl.setContentsMargins(10, 4, 10, 4)
        fl.setSpacing(8)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by symbol  (press / to focus)")
        self.filter_edit.setFont(mono_font(12))
        self.filter_edit.setFixedHeight(22)
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._on_filter)
        fl.addWidget(self.filter_edit, 1)

        self.count_label = QLabel("0")
        self.count_label.setFont(mono_font(11, 500))
        self.count_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        fl.addWidget(self.count_label)

        lay.addWidget(filter_row)

        # Table
        self._model = _MarketModel()
        self._proxy = _SortProxy()
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(Qt.UserRole)
        self._proxy.setFilterKeyColumn(-1)

        self.table = QTableView()
        self.table.setModel(self._proxy)
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        self.table.setFocusPolicy(Qt.StrongFocus)
        for i, (_k, _h, w, _a) in enumerate(COLUMNS):
            self.table.setColumnWidth(i, w)

        # Sparkline delegate
        self.table.setItemDelegateForColumn(COL_INDEX["spark"], _SparkDelegate(self))

        self.table.doubleClicked.connect(self._activate_row)
        self.table.activated.connect(self._activate_row)
        self.table.clicked.connect(self._on_clicked)

        # initial sort by % descending
        self.table.sortByColumn(COL_INDEX["pct"], Qt.DescendingOrder)

        lay.addWidget(self.table, 1)

    # data flow
    def refresh(self):
        snap = self._market.snapshot()
        self._model.set_rows(snap.quotes)
        self.count_label.setText(
            f"{len(snap.quotes)} symbols   ▲ {snap.advancers}   ▼ {snap.decliners}   = {snap.unchanged}"
        )
        self.header.set_subtitle(f"{len(snap.quotes)} symbols")
        if not self._spark_loaded:
            QTimer.singleShot(100, self._load_sparks)

    def _load_sparks(self):
        """Populate 30-bar close sparklines off the main thread's hot path."""
        self._spark_loaded = True
        out: dict[str, np.ndarray] = {}
        # For perf, load sparks only for visible rows count up to 500.
        snap = self._market.snapshot()
        syms = [q.symbol for q in snap.quotes][:500]
        for s in syms:
            h = self._market.history(s)
            if not h.empty:
                out[s] = h.close[-30:] if h.close.size >= 30 else h.close.copy()
        self._model.set_sparks(out)

    def _on_filter(self, text: str):
        self._proxy.setFilterFixedString(text.strip())
        self.count_label.setText(f"{self._proxy.rowCount()} match")

    def _activate_row(self, index: QModelIndex):
        if not index.isValid():
            return
        src_row = self._proxy.mapToSource(index).row()
        sym = self._model.row_symbol(src_row)
        if sym:
            self._link.set_symbol(sym)

    def _on_clicked(self, index: QModelIndex):
        # Single click updates the link context too (lightweight live preview)
        self._activate_row(index)

    def focus_filter(self):
        self.filter_edit.setFocus()
        self.filter_edit.selectAll()
