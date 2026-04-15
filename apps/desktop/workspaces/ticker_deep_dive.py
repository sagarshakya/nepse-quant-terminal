"""Ticker Deep Dive — single-symbol chart + rich stats / fundamentals panel."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QRectF, QSize
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QHeaderView, QLabel, QSizePolicy, QSplitter, QTableView,
    QVBoxLayout, QWidget,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, TEXT, TEXT_SECONDARY, TEXT_MUTED, GAIN, LOSS, ACCENT, WARN,
    ui_font, mono_font,
)

WARN_COLOR = WARN
from apps.desktop.utils import fmt_number, fmt_signed, fmt_signed_pct, fmt_volume
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.chart_pane import ChartPane
from backend.core.services import MarketService


# =============================================================================
# Formatting helpers
# =============================================================================
def _fmt_npr(v: Optional[float]) -> str:
    if v is None or v == 0:
        return "—"
    v = float(v)
    if abs(v) >= 1e9: return f"NPR {v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"NPR {v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"NPR {v/1e3:.1f}K"
    return f"NPR {v:,.0f}"


def _fmt_ratio(v: Optional[float], suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}{suffix}"
    except Exception:
        return "—"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "—"


# =============================================================================
# Visual primitives
# =============================================================================
class _StatRow(QFrame):
    """label on left, value on right — with subtle bottom hairline for density."""
    def __init__(self, label: str, min_label_w: int = 78):
        super().__init__()
        self.setMinimumHeight(22)
        self.setStyleSheet(
            "_StatRow { border: none; border-bottom: 1px solid rgba(255,255,255,0.035); }"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(8)
        self._label = QLabel(label.upper())
        self._label.setFont(ui_font(9, 600))
        self._label.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1.2px;")
        self._label.setMinimumWidth(min_label_w)
        self._value = QLabel("—")
        self._value.setFont(mono_font(12, 500))
        self._value.setStyleSheet(f"color: {TEXT};")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self._label)
        lay.addStretch(1)
        lay.addWidget(self._value)

    def set_value(self, value: str, color: Optional[str] = None):
        self._value.setText(value)
        self._value.setStyleSheet(f"color: {color or TEXT};")


class _Card(QFrame):
    """Borderless section: title + hairline underline + content."""
    def __init__(self, title: str, accent: str = ACCENT):
        super().__init__()
        self.setStyleSheet("_Card { background: transparent; border: none; }")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)

        ttl = QLabel(title.upper())
        ttl.setFont(ui_font(9, 700))
        ttl.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 1.8px;")
        self._lay.addWidget(ttl)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER}; border: none;")
        self._lay.addWidget(sep)

    def add_widget(self, w: QWidget):
        self._lay.addWidget(w)

    def add_rows(self, rows: list[QWidget]):
        for r in rows:
            self._lay.addWidget(r)
        self._lay.addStretch(1)


class _Sparkline(QFrame):
    """Inline sparkline — last N closes, filled area under curve."""
    def __init__(self):
        super().__init__()
        self._vals = np.array([], dtype=np.float64)
        self.setMinimumHeight(32)
        self.setMaximumHeight(40)

    def set_data(self, vals):
        self._vals = np.asarray(vals, dtype=np.float64)
        self.update()

    def paintEvent(self, event):
        if self._vals.size < 2:
            return
        from PySide6.QtGui import QPainterPath, QPolygonF
        from PySide6.QtCore import QPointF as _QP
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        w, h = rect.width(), rect.height()
        vlo = float(np.nanmin(self._vals)); vhi = float(np.nanmax(self._vals))
        rng = max(vhi - vlo, 1e-9)
        xs = np.linspace(0, w, self._vals.size)
        ys = h - ((self._vals - vlo) / rng) * (h - 6) - 3
        up = self._vals[-1] >= self._vals[0]
        line_col = QColor(GAIN if up else LOSS)
        fill_col = QColor(GAIN if up else LOSS); fill_col.setAlpha(40)
        # fill polygon
        poly = QPolygonF()
        poly.append(_QP(xs[0], h))
        for i in range(len(xs)):
            poly.append(_QP(float(xs[i]), float(ys[i])))
        poly.append(_QP(xs[-1], h))
        p.setPen(Qt.NoPen)
        p.setBrush(fill_col)
        p.drawPolygon(poly)
        # line
        pen = QPen(line_col, 1.4); pen.setCosmetic(True)
        p.setPen(pen)
        path = QPainterPath()
        path.moveTo(float(xs[0]), float(ys[0]))
        for i in range(1, len(xs)):
            path.lineTo(float(xs[i]), float(ys[i]))
        p.drawPath(path)


class _HeroCard(QFrame):
    """Sector · symbol · price · change · sparkline — borderless, top-aligned."""
    _accent_color: str = ACCENT

    def __init__(self):
        super().__init__()
        self.setStyleSheet("_HeroCard { background: transparent; border: none; }")
        bl = QVBoxLayout(self)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)

        self.sector = QLabel("—")
        self.sector.setFont(ui_font(9, 700))
        self.sector.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 1.8px;")
        bl.addWidget(self.sector)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER}; border: none;")
        bl.addWidget(sep)

        sym_row = QHBoxLayout()
        sym_row.setSpacing(6)
        sym_row.setContentsMargins(0, 2, 0, 0)
        self.symbol = QLabel("—")
        self.symbol.setFont(ui_font(18, 700))
        self.symbol.setStyleSheet(f"color: {TEXT}; letter-spacing: 0.8px;")
        self.arrow = QLabel("")
        self.arrow.setFont(ui_font(13, 700))
        sym_row.addWidget(self.symbol)
        sym_row.addWidget(self.arrow)
        sym_row.addStretch(1)
        bl.addLayout(sym_row)

        self.last = QLabel("—")
        self.last.setFont(mono_font(30, 700))
        self.last.setStyleSheet(f"color: {TEXT};")
        bl.addWidget(self.last)

        self.chg = QLabel("")
        self.chg.setFont(mono_font(12, 600))
        bl.addWidget(self.chg)

        self.spark = _Sparkline()
        self.spark.setMinimumHeight(30)
        self.spark.setMaximumHeight(40)
        bl.addWidget(self.spark)

        bl.addStretch(1)

        self.as_of = QLabel("")
        self.as_of.setFont(mono_font(9, 500))
        self.as_of.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 0.8px;")
        bl.addWidget(self.as_of)

    def set_hero(self, symbol: str, sector: str, last: float, chg: float,
                 chg_pct: float, as_of: str, spark_vals=None):
        self.sector.setText(sector.upper() if sector else "—")
        self.symbol.setText(symbol)
        self.last.setText(fmt_number(last))
        self.chg.setText(f"{fmt_signed(chg)}   {fmt_signed_pct(chg_pct)}")
        up = chg >= 0
        col = GAIN if up else LOSS
        self.chg.setStyleSheet(f"color: {col};")
        self.arrow.setText("▲" if up else "▼")
        self.arrow.setStyleSheet(f"color: {col};")
        self.last.setStyleSheet(f"color: {TEXT};")
        self.as_of.setText(as_of.upper())
        if spark_vals is not None:
            self.spark.set_data(spark_vals)


# ---------------------------------------------------------------------------
# Custom visual widgets
# ---------------------------------------------------------------------------
class _ReturnBar(QFrame):
    """Label | bipolar horizontal bar | value.

    Bar grows right for positive returns (green), left for negative (red),
    rooted at a vertical center-line.
    """
    def __init__(self, label: str):
        super().__init__()
        self.setMinimumHeight(22)
        self.setStyleSheet(
            "_ReturnBar { border: none; border-bottom: 1px solid rgba(255,255,255,0.035); }"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(10)

        self._label = QLabel(label.upper())
        self._label.setFont(ui_font(9, 700))
        self._label.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1.2px;")
        self._label.setFixedWidth(32)

        self._bar = _BarWidget()
        self._bar.setMinimumWidth(80)
        self._bar.setMinimumHeight(16)

        self._value = QLabel("—")
        self._value.setFont(mono_font(11, 600))
        self._value.setStyleSheet(f"color: {TEXT_MUTED};")
        self._value.setFixedWidth(62)
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(self._label)
        lay.addWidget(self._bar, 1)
        lay.addWidget(self._value)

    def set_value(self, pct: Optional[float], max_abs: float = 20.0):
        if pct is None:
            self._value.setText("—")
            self._value.setStyleSheet(f"color: {TEXT_MUTED};")
            self._bar.set_pct(None, max_abs=max_abs)
            return
        col = GAIN if pct >= 0 else LOSS
        sign = "+" if pct >= 0 else ""
        self._value.setText(f"{sign}{pct:.2f}%")
        self._value.setStyleSheet(f"color: {col};")
        self._bar.set_pct(pct, max_abs=max_abs)


class _BarWidget(QFrame):
    """Flat bipolar horizontal bar — square edges, tick marks at ±50%."""
    def __init__(self):
        super().__init__()
        self._pct: Optional[float] = None
        self._max_abs: float = 20.0
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)

    def set_pct(self, pct: Optional[float], max_abs: float = 20.0):
        self._pct = pct
        self._max_abs = max(0.5, max_abs)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect()
        w, h = rect.width(), rect.height()
        cx = int(w / 2.0)
        bar_h = 8
        y = int((h - bar_h) / 2.0)

        # gutter rail
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1A1D24"))
        p.drawRect(QRectF(0, y, w, bar_h))

        # centre line (zero reference) — subtle
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(cx, y - 1, cx, y + bar_h + 1)

        if self._pct is None:
            return

        pct = max(-self._max_abs, min(self._max_abs, self._pct))
        half = w / 2.0
        frac = abs(pct) / self._max_abs
        bar_w = max(frac * half, 2.0)
        col = QColor(GAIN if pct >= 0 else LOSS)

        p.setPen(Qt.NoPen)
        p.setBrush(col)
        if pct >= 0:
            p.drawRect(QRectF(cx, y, bar_w, bar_h))
        else:
            p.drawRect(QRectF(cx - bar_w, y, bar_w, bar_h))


class _RangeGauge(QFrame):
    """52W range: flat track, colored fill to current position, vertical marker."""
    def __init__(self):
        super().__init__()
        self._lo: Optional[float] = None
        self._hi: Optional[float] = None
        self._last: Optional[float] = None
        self.setMinimumHeight(52)
        self.setMaximumHeight(56)

    def set_range(self, lo: float, hi: float, last: float):
        self._lo = lo; self._hi = hi; self._last = last
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        if not self._lo or not self._hi or self._hi <= self._lo:
            return

        pad = 4
        track_y = 22
        track_h = 5
        track_w = w - pad * 2

        # top row: low | high labels
        p.setFont(mono_font(10, 600))
        p.setPen(QColor(LOSS))
        p.drawText(QRectF(pad, 2, 120, 16),
                   Qt.AlignLeft | Qt.AlignVCenter, fmt_number(self._lo))
        p.setPen(QColor(GAIN))
        p.drawText(QRectF(w - pad - 120, 2, 120, 16),
                   Qt.AlignRight | Qt.AlignVCenter, fmt_number(self._hi))

        # flat dark rail
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1A1D24"))
        p.drawRect(QRectF(pad, track_y, track_w, track_h))

        rng = (self._hi - self._lo) or 1e-9
        frac = (self._last - self._lo) / rng
        frac = max(0.0, min(1.0, frac))
        fill_w = frac * track_w

        # color fill ramps with position (low → LOSS, mid → WARN, high → GAIN)
        if frac < 0.33:
            fill_col = QColor(LOSS)
        elif frac > 0.67:
            fill_col = QColor(GAIN)
        else:
            fill_col = QColor(WARN)
        p.setBrush(fill_col)
        p.drawRect(QRectF(pad, track_y, fill_w, track_h))

        # vertical marker line through the rail
        x = int(pad + fill_w)
        p.setPen(QPen(QColor(TEXT), 2))
        p.drawLine(x, track_y - 5, x, track_y + track_h + 5)

        # current price label below marker
        p.setFont(mono_font(11, 700))
        p.setPen(QColor(TEXT))
        label = fmt_number(self._last)
        lbl_w = 90
        lx = max(pad, min(x - lbl_w / 2, w - pad - lbl_w))
        p.drawText(QRectF(lx, track_y + track_h + 8, lbl_w, 16),
                   Qt.AlignCenter, label)


class _DayRangeBar(QFrame):
    """Intraday span: flat rail, O-C body in gain/loss color, close marker."""
    def __init__(self):
        super().__init__()
        self._o = self._h = self._l = self._c = None
        self.setMinimumHeight(40)
        self.setMaximumHeight(44)

    def set_ohlc(self, o: float, h: float, lo: float, c: float):
        self._o, self._h, self._l, self._c = o, h, lo, c
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        if self._h is None or self._l is None:
            return
        rect = self.rect()
        w, h = rect.width(), rect.height()
        pad = 4
        track_y = 20
        track_h = 5
        track_w = w - pad * 2

        # top: L / H labels
        p.setFont(mono_font(9, 600))
        p.setPen(QColor(TEXT_MUTED))
        p.drawText(QRectF(pad, 2, 120, 14),
                   Qt.AlignLeft | Qt.AlignVCenter, f"L  {fmt_number(self._l)}")
        p.drawText(QRectF(w - pad - 120, 2, 120, 14),
                   Qt.AlignRight | Qt.AlignVCenter, f"H  {fmt_number(self._h)}")

        # flat rail
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1A1D24"))
        p.drawRect(QRectF(pad, track_y, track_w, track_h))

        if self._h <= self._l or self._o is None or self._c is None:
            # draw a neutral centered tick and exit
            cx_pos = int(pad + track_w / 2)
            p.setPen(QPen(QColor(TEXT), 2))
            p.drawLine(cx_pos, track_y - 5, cx_pos, track_y + track_h + 5)
            return

        rng = (self._h - self._l) or 1e-9
        ox = pad + (self._o - self._l) / rng * track_w
        cx = pad + (self._c - self._l) / rng * track_w
        lo_x, hi_x = min(ox, cx), max(ox, cx)

        bar_col = QColor(GAIN if self._c >= self._o else LOSS)
        p.setBrush(bar_col)
        p.drawRect(QRectF(lo_x, track_y, max(hi_x - lo_x, 2), track_h))

        # open tick (thin)
        p.setPen(QPen(QColor(TEXT_MUTED), 1))
        p.drawLine(int(ox), track_y - 3, int(ox), track_y + track_h + 3)
        # close marker (bold, white)
        p.setPen(QPen(QColor(TEXT), 2))
        p.drawLine(int(cx), track_y - 5, int(cx), track_y + track_h + 5)


# =============================================================================
# Quarterly financials + corporate actions tables
# =============================================================================
_QTR_COLS = [
    ("period",     "PERIOD",     90, "left"),
    ("revenue",    "REVENUE",   130, "right"),
    ("net_profit", "NET PROFIT",130, "right"),
    ("eps",        "EPS",        80, "right"),
    ("bvps",       "BVPS",       80, "right"),
    ("announce",   "ANNOUNCED", 110, "left"),
]


class _QtrModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list[dict] = []
        self._f = mono_font(11)
        self._fb = mono_font(11, 500)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_QTR_COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return _QTR_COLS[section][1]
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            a = _QTR_COLS[section][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        return None

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid(): return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows): return None
        row = self._rows[r]
        key = _QTR_COLS[c][0]
        if role == Qt.DisplayRole:
            if key == "period":
                fy = row["fiscal_year"] or "—"
                q = row["quarter"]
                return f"{fy}  Q{q}" if q else fy
            if key == "revenue":    return _fmt_npr(row["revenue"])
            if key == "net_profit": return _fmt_npr(row["net_profit"])
            if key == "eps":        return _fmt_ratio(row["eps"])
            if key == "bvps":       return _fmt_ratio(row["book_value"])
            if key == "announce":   return row["announcement"] or "—"
        if role == Qt.TextAlignmentRole:
            a = _QTR_COLS[c][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            if key == "period":  return QColor(TEXT)
            if key == "net_profit":
                v = row["net_profit"]
                if v is not None:
                    return QColor(GAIN if v >= 0 else LOSS)
            if key == "announce": return QColor(TEXT_MUTED)
            return QColor(TEXT_SECONDARY)
        if role == Qt.FontRole:
            return self._fb if key == "period" else self._f
        return None

    def set_rows(self, rows: list[dict]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


_CORP_COLS = [
    ("book_close", "BOOK CLOSE", 110, "left"),
    ("fy",         "FY",          80, "left"),
    ("cash",       "CASH %",      80, "right"),
    ("bonus",      "BONUS %",     80, "right"),
    ("right",      "RIGHT",       80, "left"),
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
            if key == "book_close": return row["book_close"] or "—"
            if key == "fy":         return row["fiscal_year"] or "—"
            if key == "cash":       return f"{row['cash']:.2f}" if row["cash"] else "—"
            if key == "bonus":      return f"{row['bonus']:.2f}" if row["bonus"] else "—"
            if key == "right":      return row["right"] or "—"
        if role == Qt.TextAlignmentRole:
            a = _CORP_COLS[c][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            if key == "book_close": return QColor(TEXT)
            if key == "cash" and row["cash"]:  return QColor(GAIN)
            if key == "bonus" and row["bonus"]: return QColor(ACCENT)
            return QColor(TEXT_SECONDARY)
        if role == Qt.FontRole:
            return self._fb if key == "book_close" else self._f
        return None

    def set_rows(self, rows: list[dict]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


def _style_table(t: QTableView, row_h: int = 26):
    t.setShowGrid(False)
    t.setAlternatingRowColors(False)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(row_h)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    t.horizontalHeader().setHighlightSections(False)
    t.horizontalHeader().setFixedHeight(26)
    t.setSelectionBehavior(QTableView.SelectRows)
    t.setSelectionMode(QTableView.SingleSelection)
    t.setEditTriggers(QTableView.NoEditTriggers)
    t.setFocusPolicy(Qt.NoFocus)
    t.setStyleSheet(f"""
        QTableView {{
            background: transparent; color: {TEXT};
            selection-background-color: {ELEVATED};
            selection-color: {TEXT}; border: none;
            gridline-color: transparent;
        }}
        QTableView::item {{ padding: 2px 10px; border: 0;
                            background: transparent; }}
        QHeaderView::section {{
            background: transparent; color: {TEXT_MUTED};
            border: none; border-bottom: 1px solid {BORDER};
            padding: 4px 10px; font-size: 10px;
            font-weight: 700; letter-spacing: 1.4px;
        }}
        QHeaderView {{ background: transparent; border: none; }}
        QScrollBar:vertical {{
            background: transparent; width: 8px;
            margin: 2px 0 2px 0; border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER}; min-height: 24px;
            border-radius: 3px; border: none;
        }}
        QScrollBar::handle:vertical:hover {{ background: {TEXT_MUTED}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            background: transparent; height: 0; border: none;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QScrollBar:horizontal {{
            background: transparent; height: 8px;
            margin: 0 2px 0 2px; border: none;
        }}
        QScrollBar::handle:horizontal {{
            background: {BORDER}; min-width: 24px;
            border-radius: 3px; border: none;
        }}
        QScrollBar::handle:horizontal:hover {{ background: {TEXT_MUTED}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            background: transparent; width: 0; border: none;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
    """)


class _TableCard(QFrame):
    """Borderless section: title + underline + QTableView."""
    def __init__(self, title: str, model: QAbstractTableModel, col_widths: list[int],
                 accent: str = ACCENT):
        super().__init__()
        self.setStyleSheet("_TableCard { background: transparent; border: none; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        ttl = QLabel(title.upper())
        ttl.setFont(ui_font(9, 700))
        ttl.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 1.8px;")
        lay.addWidget(ttl)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER}; border: none;")
        lay.addWidget(sep)

        self.table = QTableView()
        self.table.setModel(model)
        _style_table(self.table)
        for i, w in enumerate(col_widths):
            self.table.setColumnWidth(i, w)
        lay.addWidget(self.table, 1)


# =============================================================================
# Stats + fundamentals panel
# =============================================================================
class _StatsPanel(QFrame):
    """Two-row composition: KPI cards on top, finance tables below."""

    def __init__(self, market: MarketService, link: LinkGroup,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._market = market
        self._link = link
        self.setStyleSheet(f"_StatsPanel {{ background: {BG}; border-top: 1px solid {BORDER}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # ---- Row 1: KPI cards with slim vertical dividers -----------------
        def _vdiv() -> QFrame:
            f = QFrame()
            f.setFixedWidth(1)
            f.setStyleSheet(f"background: {BORDER}; border: none;")
            return f

        row1 = QHBoxLayout()
        row1.setSpacing(18)

        self.hero = _HeroCard()
        self.hero.setMinimumWidth(200)
        row1.addWidget(self.hero, 3)
        row1.addWidget(_vdiv())

        snap = _Card("Session")
        self._day_bar = _DayRangeBar()
        snap.add_widget(self._day_bar)
        self._open = _StatRow("Open")
        self._prev = _StatRow("Prev Close")
        self._vol  = _StatRow("Volume")
        self._avgv = _StatRow("Avg 20D")
        snap.add_rows([self._open, self._prev, self._vol, self._avgv])
        row1.addWidget(snap, 3)
        row1.addWidget(_vdiv())

        rets = _Card("Returns", accent=GAIN)
        self._r1m  = _ReturnBar("1M")
        self._r3m  = _ReturnBar("3M")
        self._r6m  = _ReturnBar("6M")
        self._r1y  = _ReturnBar("1Y")
        self._rytd = _ReturnBar("YTD")
        rets.add_rows([self._r1m, self._r3m, self._r6m, self._r1y, self._rytd])
        row1.addWidget(rets, 4)
        row1.addWidget(_vdiv())

        rng = _Card("52-Week Range", accent=WARN_COLOR)
        self._range_gauge = _RangeGauge()
        rng.add_widget(self._range_gauge)
        self._from_hi = _StatRow("From High")
        self._vola    = _StatRow("Volatility")
        self._mcap    = _StatRow("Mkt Cap")
        rng.add_rows([self._from_hi, self._vola, self._mcap])
        row1.addWidget(rng, 4)
        row1.addWidget(_vdiv())

        val = _Card("Valuation", accent=ACCENT)
        self._pe       = _StatRow("P / E")
        self._pb       = _StatRow("P / B")
        self._eps_f    = _StatRow("EPS")
        self._bvps_f   = _StatRow("BVPS")
        self._roe      = _StatRow("ROE")
        self._divy     = _StatRow("Div Yield")
        val.add_rows([self._pe, self._pb, self._eps_f, self._bvps_f, self._roe, self._divy])
        row1.addWidget(val, 3)

        outer.addLayout(row1, 4)

        hr = QFrame()
        hr.setFixedHeight(1)
        hr.setStyleSheet(f"background: {BORDER}; border: none;")
        outer.addWidget(hr)

        # ---- Row 2: tables ------------------------------------------------
        row2 = QHBoxLayout()
        row2.setSpacing(18)

        self._qtr_model = _QtrModel()
        self._corp_model = _CorpModel()
        qtr_card = _TableCard(
            "Quarterly Financials",
            self._qtr_model,
            [c[2] for c in _QTR_COLS],
            accent=GAIN,
        )
        corp_card = _TableCard(
            "Corporate Actions",
            self._corp_model,
            [c[2] for c in _CORP_COLS],
            accent=WARN_COLOR,
        )
        row2.addWidget(qtr_card, 3)
        vdiv2 = QFrame()
        vdiv2.setFixedWidth(1)
        vdiv2.setStyleSheet(f"background: {BORDER}; border: none;")
        row2.addWidget(vdiv2)
        row2.addWidget(corp_card, 2)

        outer.addLayout(row2, 5)

        link.symbol_changed.connect(self.set_symbol)

    # ---- data ---------------------------------------------------------------
    def set_symbol(self, symbol: str):
        if not symbol:
            return
        h = self._market.history(symbol)
        fund = self._market.fundamentals_latest(symbol) or {}
        sector = (fund.get("sector") or "").strip()

        if h.empty:
            self.hero.set_hero(symbol, sector, 0.0, 0.0, 0.0, "no data", None)
            for r in (self._open, self._prev, self._vol, self._avgv,
                      self._from_hi, self._vola, self._mcap,
                      self._pe, self._pb, self._eps_f, self._bvps_f, self._roe, self._divy):
                r.set_value("—", TEXT_MUTED)
            for b in (self._r1m, self._r3m, self._r6m, self._r1y, self._rytd):
                b.set_value(None)
            self._day_bar.setVisible(False)
            self._range_gauge.set_range(0, 0, 0)
            self._qtr_model.set_rows([])
            self._corp_model.set_rows([])
            return

        c = h.close; hi = h.high; lo = h.low; o = h.open; v = h.volume; d = h.dates
        last = float(c[-1])
        prev = float(c[-2]) if c.size > 1 else last
        chg = last - prev
        chg_pct = (chg / prev * 100.0) if prev else 0.0

        spark_n = min(90, c.size)
        self.hero.set_hero(symbol, sector, last, chg, chg_pct, str(d[-1]),
                           c[-spark_n:] if spark_n > 1 else None)

        # Session card
        has_range = float(hi[-1]) > float(lo[-1])
        self._day_bar.setVisible(has_range)
        if has_range:
            self._day_bar.set_ohlc(float(o[-1]), float(hi[-1]), float(lo[-1]), last)
        self._open.set_value(fmt_number(float(o[-1])))
        self._prev.set_value(fmt_number(prev))
        self._vol.set_value(fmt_volume(float(v[-1])))
        avg_v = float(np.nanmean(v[-20:])) if v.size >= 20 else float(np.nanmean(v))
        self._avgv.set_value(fmt_volume(avg_v))

        # Returns
        def ret_n(n: int) -> Optional[float]:
            if c.size <= n: return None
            base = float(c[-n - 1])
            if base == 0: return None
            return (last / base - 1.0) * 100.0

        r1m = ret_n(22); r3m = ret_n(66); r6m = ret_n(132); r1y = ret_n(252)
        try:
            last_year = int(str(d[-1])[:4])
            ytd_idx = int(np.argmax(d >= np.datetime64(f"{last_year}-01-01")))
            if ytd_idx > 0 and c[ytd_idx] != 0:
                ytd = (last / float(c[ytd_idx]) - 1.0) * 100.0
            else:
                ytd = None
        except Exception:
            ytd = None

        vals = [v for v in (r1m, r3m, r6m, r1y, ytd) if v is not None]
        max_abs = max((abs(v) for v in vals), default=10.0)
        max_abs = max(max_abs, 5.0)
        self._r1m.set_value(r1m, max_abs)
        self._r3m.set_value(r3m, max_abs)
        self._r6m.set_value(r6m, max_abs)
        self._r1y.set_value(r1y, max_abs)
        self._rytd.set_value(ytd, max_abs)

        # 52W gauge + misc
        window = min(252, c.size)
        hi52 = float(np.nanmax(hi[-window:]))
        lo52 = float(np.nanmin(lo[-window:]))
        from_hi = (last / hi52 - 1.0) * 100.0 if hi52 else 0.0
        self._range_gauge.set_range(lo52, hi52, last)
        self._from_hi.set_value(fmt_signed_pct(from_hi), GAIN if from_hi >= 0 else LOSS)

        if c.size >= 21:
            rets = np.diff(c[-21:]) / c[-21:-1]
            vola = float(np.nanstd(rets) * np.sqrt(252) * 100.0)
            self._vola.set_value(f"{vola:.2f}%")
        else:
            self._vola.set_value("—")

        self._mcap.set_value(_fmt_npr(fund.get("market_cap")))

        # Valuation
        self._pe.set_value(_fmt_ratio(fund.get("pe")))
        self._pb.set_value(_fmt_ratio(fund.get("pb")))
        self._eps_f.set_value(_fmt_ratio(fund.get("eps")))
        self._bvps_f.set_value(_fmt_ratio(fund.get("bvps")))
        roe = fund.get("roe")
        if roe is None:
            self._roe.set_value("—")
        else:
            roe_pct = float(roe) * 100 if abs(float(roe)) < 5 else float(roe)
            self._roe.set_value(f"{roe_pct:.2f}%")
        dy = fund.get("div_yield")
        if dy is None:
            self._divy.set_value("—")
        else:
            dy_pct = float(dy) * 100 if abs(float(dy)) < 1 else float(dy)
            self._divy.set_value(f"{dy_pct:.2f}%")

        # Tables
        try:
            qtr = self._market.quarterly_earnings(symbol, limit=8)
            qtr = [q for q in qtr if any(
                q.get(k) not in (None, 0) for k in
                ("revenue", "net_profit", "eps", "book_value")
            )]
            self._qtr_model.set_rows(qtr)
        except Exception:
            self._qtr_model.set_rows([])
        try:
            self._corp_model.set_rows(
                self._market.corporate_actions_for_symbol(symbol, limit=10)
            )
        except Exception:
            self._corp_model.set_rows([])


# =============================================================================
# Workspace
# =============================================================================
class TickerDeepDive(QWidget):
    def __init__(self, market: MarketService, link: LinkGroup,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._market = market
        self._link = link

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.chart = ChartPane(market, link, default_window_bars=180)
        if hasattr(self.chart, "header"):
            self.chart.header.hide()

        self.stats = _StatsPanel(market, link)

        v_split = QSplitter(Qt.Vertical)
        v_split.setHandleWidth(1)
        v_split.addWidget(self.chart)
        v_split.addWidget(self.stats)
        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 1)
        v_split.setSizes([500, 520])
        lay.addWidget(v_split, 1)
