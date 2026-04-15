"""Dashboard — default landing workspace, TUI-inspired 4-quadrant layout.

    ┌───────────────────────────────┬────────────────────────────────────┐
    │  SIGNALS                      │  ACTIVE  — top 200 by volume         │
    │  symbol last type as regime s │  symbol sector vol change% trend30d  │
    ├───────────────────────────────┼────────────────────────────────────┤
    │  CORPORATE ACTIONS (30 days)  │  SECTOR PERFORMANCE                  │
    │  symbol book close cash bonus │  horizontal bar chart                │
    └───────────────────────────────┴────────────────────────────────────┘
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QTimer, QPointF,
    QRectF, QSize,
)
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QHeaderView, QLabel, QSplitter, QStyledItemDelegate,
    QStyleOptionViewItem, QTableView, QVBoxLayout, QWidget,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, TEXT, TEXT_SECONDARY, TEXT_MUTED,
    GAIN, LOSS, ACCENT, WARN, mono_font, ui_font,
)
from apps.desktop.utils import fmt_number, fmt_signed_pct, fmt_volume
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.pane_header import PaneHeader
from backend.core.services import MarketService, SignalService


# ---------------------------------------------------------------------------
# SIGNALS table
# ---------------------------------------------------------------------------
_SIG_COLS = [
    ("symbol", "SYMBOL", 78, "left"),
    ("last",   "LAST",   72, "right"),
    ("name",   "TYPE",  140, "left"),
    ("as_of",  "AS",     84, "left"),
    ("regime", "REGIME", 70, "left"),
    ("score",  "STR",    60, "right"),
]


class _SignalsModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list = []   # list of (Signal, last_price)
        self._f = mono_font(11)
        self._fb = mono_font(11, 500)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_SIG_COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return _SIG_COLS[section][1]
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            a = _SIG_COLS[section][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        return None

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid(): return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows): return None
        sig, last = self._rows[r]
        key = _SIG_COLS[c][0]
        if role == Qt.DisplayRole:
            if key == "symbol": return sig.symbol
            if key == "last":   return fmt_number(last) if last else "—"
            if key == "name":   return sig.name
            if key == "as_of":  return sig.as_of.isoformat()
            if key == "regime": return sig.regime or "—"
            if key == "score":  return f"{sig.score:.2f}"
        if role == Qt.TextAlignmentRole:
            a = _SIG_COLS[c][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            if key == "symbol": return QColor(TEXT)
            if key == "name":   return QColor(ACCENT)
            if key == "regime":
                reg = (sig.regime or "").lower()
                if reg == "bull":    return QColor(GAIN)
                if reg == "bear":    return QColor(LOSS)
                return QColor(TEXT_SECONDARY)
            if key == "score":
                return QColor(GAIN if sig.score >= 0 else LOSS)
            return QColor(TEXT_SECONDARY)
        if role == Qt.FontRole:
            return self._fb if key == "symbol" else self._f
        if role == Qt.UserRole:
            return sig.symbol
        return None

    def set_rows(self, rows: list):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


# ---------------------------------------------------------------------------
# ACTIVE (top-volume) table with sparklines
# ---------------------------------------------------------------------------
_ACT_COLS = [
    ("symbol", "SYMBOL",  78, "left"),
    ("sector", "SECTOR", 150, "left"),
    ("last",   "LAST",    80, "right"),
    ("pct",    "CHG%",    74, "right"),
    ("volume", "VOL",     84, "right"),
    ("spark",  "TREND",  100, "center"),
]


class _ActiveModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list = []   # (Quote, sector, trend_arr)
        self._f = mono_font(11)
        self._fb = mono_font(11, 500)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_ACT_COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return _ACT_COLS[section][1]
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            a = _ACT_COLS[section][3]
            if a == "left":   return int(Qt.AlignLeft  | Qt.AlignVCenter)
            if a == "center": return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid(): return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows): return None
        q, sector, trend = self._rows[r]
        key = _ACT_COLS[c][0]
        if role == Qt.DisplayRole:
            if key == "symbol": return q.symbol
            if key == "sector": return sector or "—"
            if key == "last":   return fmt_number(q.last)
            if key == "pct":    return fmt_signed_pct(q.change_pct)
            if key == "volume": return fmt_volume(q.volume)
            if key == "spark":  return ""
        if role == Qt.TextAlignmentRole:
            a = _ACT_COLS[c][3]
            if a == "left":   return int(Qt.AlignLeft  | Qt.AlignVCenter)
            if a == "center": return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            if key == "symbol": return QColor(TEXT)
            if key == "sector": return QColor(TEXT_SECONDARY)
            if key == "pct":
                if q.change_pct > 0: return QColor(GAIN)
                if q.change_pct < 0: return QColor(LOSS)
                return QColor(TEXT_MUTED)
            if key == "volume": return QColor(TEXT_SECONDARY)
            return QColor(TEXT)
        if role == Qt.FontRole:
            return self._fb if key == "symbol" else self._f
        if role == Qt.UserRole:
            return q.symbol
        if role == Qt.UserRole + 1 and key == "spark":
            return trend
        return None

    def set_rows(self, rows: list):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


class _SparkDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        super().paint(painter, option, index)
        spark = index.data(Qt.UserRole + 1)
        if spark is None or len(spark) < 2:
            return
        rect = option.rect.adjusted(6, 4, -6, -4)
        arr = np.asarray(spark, dtype=np.float64)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi == lo:
            return
        w, h = rect.width(), rect.height()
        xs = np.linspace(rect.left(), rect.right(), arr.size)
        ys = rect.bottom() - (arr - lo) / (hi - lo) * h
        last_chg = arr[-1] - arr[0]
        color = QColor(GAIN) if last_chg >= 0 else QColor(LOSS)
        pen = QPen(color); pen.setWidthF(1.25); pen.setCosmetic(True)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(pen)
        for i in range(arr.size - 1):
            painter.drawLine(QPointF(xs[i], ys[i]), QPointF(xs[i + 1], ys[i + 1]))
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(Qt.NoPen))
        painter.drawEllipse(QPointF(xs[-1], ys[-1]), 2.0, 2.0)
        painter.restore()


# ---------------------------------------------------------------------------
# Corporate actions table
# ---------------------------------------------------------------------------
_CORP_COLS = [
    ("symbol",     "SYMBOL",      80, "left"),
    ("book_close", "BOOK CLOSE", 110, "left"),
    ("cash",       "CASH %",      70, "right"),
    ("bonus",      "BONUS %",     70, "right"),
    ("right",      "RIGHT",       64, "left"),
    ("description","DETAILS",    400, "left"),
]


class _CorpModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list[dict] = []
        self._f = mono_font(11)
        self._fb = mono_font(11, 500)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_CORP_COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return _CORP_COLS[section][1]
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            a = _CORP_COLS[section][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        return None

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid(): return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows): return None
        row = self._rows[r]
        key = _CORP_COLS[c][0]
        if role == Qt.DisplayRole:
            if key == "cash":   return f"{row['cash']:.2f}"  if row['cash']  else "—"
            if key == "bonus":  return f"{row['bonus']:.2f}" if row['bonus'] else "—"
            if key == "description":
                return row["description"][:120]
            return row.get(key, "")
        if role == Qt.TextAlignmentRole:
            a = _CORP_COLS[c][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            if key == "symbol":     return QColor(TEXT)
            if key == "book_close": return QColor(WARN)
            if key == "cash" and row["cash"]:  return QColor(GAIN)
            if key == "bonus" and row["bonus"]: return QColor(ACCENT)
            return QColor(TEXT_SECONDARY)
        if role == Qt.FontRole:
            return self._fb if key == "symbol" else self._f
        if role == Qt.UserRole:
            return row["symbol"]
        return None

    def set_rows(self, rows: list[dict]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


# ---------------------------------------------------------------------------
# Sector performance horizontal bar chart
# ---------------------------------------------------------------------------
class _SectorBars(QFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows: list[tuple[str, float, int]] = []  # name, avg_pct, n
        self.setStyleSheet(f"_SectorBars {{ background: {PANE}; }}")

    def set_rows(self, rows: list[tuple[str, float, int]]):
        self._rows = rows
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(400, max(200, len(self._rows) * 22 + 10))

    def minimumSizeHint(self) -> QSize:
        return QSize(300, max(180, len(self._rows) * 22 + 10))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(PANE))
        if not self._rows:
            p.setPen(QColor(TEXT_MUTED))
            p.setFont(ui_font(11))
            p.drawText(self.rect(), Qt.AlignCenter, "no sector data")
            return

        margin_l = 150
        margin_r = 60
        row_h = 22
        top = 6
        avail_w = self.width() - margin_l - margin_r
        if avail_w < 60:
            return

        # Determine scale
        max_abs = max(abs(r[1]) for r in self._rows) or 1.0
        max_abs = max(max_abs, 1.0)
        center_x = margin_l + avail_w / 2.0

        p.setFont(mono_font(11))
        # zero axis line
        p.setPen(QPen(QColor(BORDER), 1, Qt.DashLine))
        p.drawLine(int(center_x), top, int(center_x), top + row_h * len(self._rows))

        p.setFont(mono_font(11))
        for i, (name, pct, n) in enumerate(self._rows):
            y = top + i * row_h
            # sector label
            p.setPen(QColor(TEXT))
            p.drawText(QRectF(8, y, margin_l - 12, row_h),
                       Qt.AlignLeft | Qt.AlignVCenter, name)
            # bar
            frac = pct / max_abs
            bar_w = frac * (avail_w / 2.0)
            color = QColor(GAIN) if pct >= 0 else QColor(LOSS)
            p.setPen(QPen(Qt.NoPen))
            p.setBrush(QBrush(color))
            if pct >= 0:
                p.drawRect(QRectF(center_x, y + 4, bar_w, row_h - 8))
            else:
                p.drawRect(QRectF(center_x + bar_w, y + 4, -bar_w, row_h - 8))
            # pct label
            p.setPen(color)
            sign = "+" if pct >= 0 else ""
            p.drawText(QRectF(margin_l + avail_w + 2, y, margin_r - 4, row_h),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{sign}{pct:.2f}%")
            # count label in parentheses faded
            p.setPen(QColor(TEXT_MUTED))
            p.drawText(QRectF(8, y, margin_l - 12, row_h),
                       Qt.AlignRight | Qt.AlignVCenter, f"({n})")


# ---------------------------------------------------------------------------
# Quadrant pane helper
# ---------------------------------------------------------------------------
def _quadrant(title: str) -> tuple[QFrame, QVBoxLayout, PaneHeader]:
    f = QFrame()
    f.setStyleSheet(f"QFrame {{ background: {PANE}; }}")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    header = PaneHeader(title, subtitle="")
    lay.addWidget(header)
    return f, lay, header


def _style_table(t: QTableView):
    t.setShowGrid(False)
    t.setAlternatingRowColors(False)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(22)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    t.horizontalHeader().setHighlightSections(False)
    t.setSelectionBehavior(QTableView.SelectRows)
    t.setSelectionMode(QTableView.SingleSelection)
    t.setEditTriggers(QTableView.NoEditTriggers)
    t.setStyleSheet(f"""
        QTableView {{ background: {PANE}; color: {TEXT};
                      selection-background-color: {ELEVATED};
                      selection-color: {TEXT}; border: none; }}
        QTableView::item {{ padding: 1px 6px; border: 0; }}
        QHeaderView::section {{ background: {PANE}; color: {TEXT_MUTED};
                                border: none; border-bottom: 1px solid {BORDER};
                                padding: 4px 6px; font-size: 10px;
                                letter-spacing: 0.8px; }}
    """)


class Dashboard(QWidget):
    def __init__(
        self,
        market: MarketService,
        signals_svc: SignalService,
        link: LinkGroup,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._market = market
        self._signals_svc = signals_svc
        self._link = link

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Main 2x2 splitter ---------------------------------------------
        outer = QSplitter(Qt.Vertical)
        outer.setHandleWidth(1)

        top_row = QSplitter(Qt.Horizontal)
        top_row.setHandleWidth(1)
        bot_row = QSplitter(Qt.Horizontal)
        bot_row.setHandleWidth(1)

        # ---- SIGNALS quadrant ---------------------------------------------
        sig_frame, sig_lay, self.sig_header = _quadrant("Signals")
        self._sig_model = _SignalsModel()
        self.sig_table = QTableView()
        self.sig_table.setModel(self._sig_model)
        _style_table(self.sig_table)
        for i, (_k, _h, w, _a) in enumerate(_SIG_COLS):
            self.sig_table.setColumnWidth(i, w)
        self.sig_table.clicked.connect(self._on_sig_click)
        sig_lay.addWidget(self.sig_table, 1)

        # ---- ACTIVE quadrant ---------------------------------------------
        act_frame, act_lay, self.act_header = _quadrant("Active — top 200 by volume")
        self._act_model = _ActiveModel()
        self.act_table = QTableView()
        self.act_table.setModel(self._act_model)
        _style_table(self.act_table)
        for i, (_k, _h, w, _a) in enumerate(_ACT_COLS):
            self.act_table.setColumnWidth(i, w)
        self.act_table.setItemDelegateForColumn(
            next(i for i, c in enumerate(_ACT_COLS) if c[0] == "spark"),
            _SparkDelegate(self.act_table),
        )
        self.act_table.clicked.connect(self._on_act_click)
        act_lay.addWidget(self.act_table, 1)

        # ---- CORPORATE ACTIONS quadrant -----------------------------------
        corp_frame, corp_lay, self.corp_header = _quadrant("Corporate Actions — next 30 days")
        self._corp_model = _CorpModel()
        self.corp_table = QTableView()
        self.corp_table.setModel(self._corp_model)
        _style_table(self.corp_table)
        for i, (_k, _h, w, _a) in enumerate(_CORP_COLS):
            self.corp_table.setColumnWidth(i, w)
        self.corp_table.clicked.connect(self._on_corp_click)
        corp_lay.addWidget(self.corp_table, 1)

        # ---- SECTOR PERFORMANCE quadrant ---------------------------------
        sect_frame, sect_lay, self.sect_header = _quadrant("Sector Performance")
        self.sect_bars = _SectorBars()
        sect_lay.addWidget(self.sect_bars, 1)

        top_row.addWidget(sig_frame)
        top_row.addWidget(act_frame)
        top_row.setStretchFactor(0, 4)
        top_row.setStretchFactor(1, 5)
        top_row.setSizes([520, 680])

        bot_row.addWidget(corp_frame)
        bot_row.addWidget(sect_frame)
        bot_row.setStretchFactor(0, 4)
        bot_row.setStretchFactor(1, 5)
        bot_row.setSizes([520, 680])

        outer.addWidget(top_row)
        outer.addWidget(bot_row)
        outer.setStretchFactor(0, 3)
        outer.setStretchFactor(1, 2)
        outer.setSizes([520, 360])

        root.addWidget(outer, 1)

        # Timer
        QTimer.singleShot(50, self._refresh)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(30_000)

    # ---- click handlers ----------------------------------------------------
    def _on_sig_click(self, idx: QModelIndex):
        sym = self._sig_model.data(idx, Qt.UserRole)
        if sym: self._link.set_symbol(sym)

    def _on_act_click(self, idx: QModelIndex):
        sym = self._act_model.data(idx, Qt.UserRole)
        if sym: self._link.set_symbol(sym)

    def _on_corp_click(self, idx: QModelIndex):
        sym = self._corp_model.data(idx, Qt.UserRole)
        if sym: self._link.set_symbol(sym)

    # ---- refresh ----------------------------------------------------------
    def _refresh(self):
        snap = self._market.snapshot()
        quotes = snap.quotes
        sector_map = self._market.sector_map()
        quotes_by_sym = {q.symbol: q for q in quotes}

        # SIGNALS
        sigs = []
        try:
            sigs = self._signals_svc.recent(limit=200)
        except Exception:
            sigs = []
        sig_rows = []
        for s in sigs[:200]:
            q = quotes_by_sym.get(s.symbol)
            sig_rows.append((s, q.last if q else 0.0))
        self._sig_model.set_rows(sig_rows)
        self.sig_header.set_subtitle(f"{len(sig_rows)} rows")

        # ACTIVE — top 200 by volume with 30d close trend
        active = sorted(quotes, key=lambda q: q.volume, reverse=True)[:200]
        act_rows = []
        # fetch histories lazily for top 50 (perf)
        for i, q in enumerate(active):
            trend = None
            if i < 50:
                try:
                    h = self._market.history(q.symbol)
                    if not h.empty:
                        trend = h.close[-30:] if h.close.size >= 30 else h.close.copy()
                except Exception:
                    trend = None
            act_rows.append((q, sector_map.get(q.symbol, ""), trend))
        self._act_model.set_rows(act_rows)
        self.act_header.set_subtitle(f"{len(active)} rows · sorted by volume")

        # CORPORATE ACTIONS
        try:
            corp = self._market.upcoming_corporate_actions(days=30, limit=60)
        except Exception:
            corp = []
        self._corp_model.set_rows(corp)
        self.corp_header.set_subtitle(f"{len(corp)} upcoming")

        # SECTOR PERFORMANCE
        by_sector: dict[str, list[float]] = defaultdict(list)
        for q in quotes:
            sec = sector_map.get(q.symbol)
            if sec:
                by_sector[sec].append(q.change_pct)
        sector_rows = []
        for sec, pcts in by_sector.items():
            if len(pcts) < 1:
                continue
            avg = float(np.mean(pcts))
            sector_rows.append((sec, avg, len(pcts)))
        sector_rows.sort(key=lambda r: r[1], reverse=True)
        self.sect_bars.set_rows(sector_rows)
        self.sect_header.set_subtitle(f"{len(sector_rows)} sectors")
