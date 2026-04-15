"""Backtests workspace — list of saved artifacts + equity curve + metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QHeaderView, QLabel, QSplitter, QTableView, QVBoxLayout,
    QWidget, QGridLayout,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, TEXT, TEXT_SECONDARY, TEXT_MUTED,
    GAIN, LOSS, ACCENT, mono_font, ui_font,
)
from apps.desktop.utils import fmt_number, fmt_signed_pct
from apps.desktop.widgets.pane_header import PaneHeader
from backend.core.services import BacktestService
from backend.core.types import BacktestSummary


# key, header, width, align
_COLS = [
    ("name",         "STRATEGY",   260, "left"),
    ("total_return", "RETURN",      92, "right"),
    ("sharpe",       "SHARPE",      70, "right"),
    ("max_dd",       "MAX DD",      80, "right"),
    ("n_trades",     "TRADES",      70, "right"),
]


class _BacktestModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list[BacktestSummary] = []
        self._font_mono = mono_font(12)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return _COLS[section][1]
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            a = _COLS[section][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        return None

    def data(self, idx: QModelIndex, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows):
            return None
        s = self._rows[r]
        key = _COLS[c][0]
        if role == Qt.DisplayRole:
            if key == "name":         return s.name
            if key == "total_return": return fmt_signed_pct(s.total_return)
            if key == "sharpe":       return f"{s.sharpe:.2f}"
            if key == "max_dd":       return f"{s.max_dd:.2f}%"
            if key == "n_trades":     return f"{s.n_trades}"
        if role == Qt.TextAlignmentRole:
            a = _COLS[c][3]
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            if key == "total_return":
                return QColor(GAIN if s.total_return >= 0 else LOSS)
            if key == "max_dd":
                return QColor(LOSS)
            if key == "sharpe":
                return QColor(GAIN if s.sharpe >= 1.0 else TEXT)
            return QColor(TEXT)
        if role == Qt.FontRole:
            return self._font_mono
        if role == Qt.UserRole:
            return s
        return None

    def set_rows(self, rows: list[BacktestSummary]):
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row: int) -> Optional[BacktestSummary]:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None


class _KPI(QFrame):
    def __init__(self, label: str):
        super().__init__()
        self.setStyleSheet(f"_KPI {{ background: {ELEVATED}; border: 1px solid {BORDER}; }}")
        self.setFixedHeight(64)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(2)
        self._l = QLabel(label.upper())
        self._l.setFont(ui_font(10, 500))
        self._l.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 0.8px;")
        self._v = QLabel("—")
        self._v.setFont(mono_font(18, 500))
        self._v.setStyleSheet(f"color: {TEXT};")
        lay.addWidget(self._l)
        lay.addWidget(self._v)

    def set_value(self, text: str, color: Optional[str] = None):
        self._v.setText(text)
        self._v.setStyleSheet(f"color: {color or TEXT};")


class Backtests(QWidget):
    def __init__(self, service: BacktestService, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._service = service

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        h_split = QSplitter(Qt.Horizontal)
        h_split.setHandleWidth(1)

        # ---- Left: artifact list ----
        left = QFrame()
        left.setStyleSheet(f"QFrame {{ background: {PANE}; }}")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self.header_list = PaneHeader("Backtests", subtitle="")
        ll.addWidget(self.header_list)

        self._model = _BacktestModel()
        self.table = QTableView()
        self.table.setModel(self._model)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        for i, (_k, _h, w, _a) in enumerate(_COLS):
            self.table.setColumnWidth(i, w)
        self.table.selectionModel = lambda: self.table.selectionModel()  # keep API
        self.table.clicked.connect(self._on_row_clicked)
        ll.addWidget(self.table, 1)

        h_split.addWidget(left)

        # ---- Right: equity curve + metrics ----
        right = QFrame()
        right.setStyleSheet(f"QFrame {{ background: {BG}; }}")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        self.header_detail = PaneHeader("Equity Curve", subtitle="")
        rl.addWidget(self.header_detail)

        # KPI row
        kpi_row = QFrame()
        kpi_row.setStyleSheet(f"QFrame {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}")
        grid = QGridLayout(kpi_row)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        self.kpi_ret    = _KPI("Total Return")
        self.kpi_sharpe = _KPI("Sharpe")
        self.kpi_dd     = _KPI("Max DD")
        self.kpi_trades = _KPI("Trades")
        self.kpi_win    = _KPI("Win Rate")
        self.kpi_score  = _KPI("Score")

        grid.addWidget(self.kpi_ret,    0, 0)
        grid.addWidget(self.kpi_sharpe, 0, 1)
        grid.addWidget(self.kpi_dd,     0, 2)
        grid.addWidget(self.kpi_trades, 0, 3)
        grid.addWidget(self.kpi_win,    0, 4)
        grid.addWidget(self.kpi_score,  0, 5)
        rl.addWidget(kpi_row)

        # Equity curve plot
        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.setBackground(BG)
        self._plot_widget.ci.setContentsMargins(0, 0, 0, 0)
        self._plot = self._plot_widget.addPlot(
            axisItems={"right": pg.AxisItem("right"), "bottom": pg.AxisItem("bottom")}
        )
        self._plot.hideAxis("left")
        self._plot.showAxis("right")
        self._plot.getAxis("right").setWidth(72)
        for side in ("bottom", "right"):
            ax = self._plot.getAxis(side)
            ax.setPen(QColor(BORDER))
            ax.setTextPen(QColor(TEXT_SECONDARY))
            ax.setStyle(tickFont=mono_font(10))
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.hideButtons()

        self._strat_curve = pg.PlotCurveItem(pen=pg.mkPen(QColor(ACCENT), width=1.6))
        self._bench_curve = pg.PlotCurveItem(pen=pg.mkPen(QColor(TEXT_MUTED), width=1.2, style=Qt.DashLine))
        self._plot.addItem(self._strat_curve)
        self._plot.addItem(self._bench_curve)

        self._legend = pg.LegendItem(offset=(10, 10), labelTextColor=QColor(TEXT_SECONDARY))
        self._legend.setParentItem(self._plot.graphicsItem())
        self._legend.addItem(self._strat_curve, "Strategy")
        self._legend.addItem(self._bench_curve, "Benchmark")

        rl.addWidget(self._plot_widget, 1)

        # Path / meta footer
        self._path_label = QLabel("")
        self._path_label.setFont(mono_font(10))
        self._path_label.setStyleSheet(f"color: {TEXT_MUTED}; padding: 6px 14px; border-top: 1px solid {BORDER};")
        rl.addWidget(self._path_label)

        h_split.addWidget(right)
        h_split.setStretchFactor(0, 2)
        h_split.setStretchFactor(1, 3)
        h_split.setSizes([520, 820])
        lay.addWidget(h_split)

        self._refresh()

    def _refresh(self):
        rows = self._service.list()
        self._model.set_rows(rows)
        self.header_list.set_subtitle(f"{len(rows)} artifacts")
        if rows:
            self.table.selectRow(0)
            self._load_artifact(rows[0])

    def _on_row_clicked(self, idx: QModelIndex):
        if not idx.isValid():
            return
        row = self._model.row_at(idx.row())
        if row is not None:
            self._load_artifact(row)

    def _load_artifact(self, summary: BacktestSummary):
        self.header_detail.set_subtitle(summary.name)
        self._path_label.setText(summary.path)

        # KPIs
        ret = summary.total_return
        self.kpi_ret.set_value(fmt_signed_pct(ret), GAIN if ret >= 0 else LOSS)
        self.kpi_sharpe.set_value(f"{summary.sharpe:.2f}", GAIN if summary.sharpe >= 1.0 else TEXT)
        self.kpi_dd.set_value(f"{summary.max_dd:.2f}%", LOSS)
        self.kpi_trades.set_value(f"{summary.n_trades}")

        # Try to load richer data from the JSON
        try:
            blob = json.loads(Path(summary.path).read_text())
        except Exception:
            blob = {}

        win = blob.get("win_rate", blob.get("metrics", {}).get("win_rate"))
        self.kpi_win.set_value(f"{float(win):.1f}%" if win is not None else "—")
        score = blob.get("score", blob.get("metrics", {}).get("score"))
        self.kpi_score.set_value(f"{float(score):.2f}" if score is not None else "—")

        # Equity curves — try several common field shapes
        strat = self._extract_curve(blob, ["equity_curve", "strategy_equity", "equity", "portfolio_equity"])
        bench = self._extract_curve(blob, ["benchmark_equity", "nepse_equity", "benchmark", "index_equity"])

        if strat is not None and strat.size > 1:
            xs = np.arange(strat.size, dtype=np.float64)
            self._strat_curve.setData(xs, strat)
        else:
            self._strat_curve.setData([], [])
        if bench is not None and bench.size > 1:
            xs = np.arange(bench.size, dtype=np.float64)
            self._bench_curve.setData(xs, bench)
        else:
            self._bench_curve.setData([], [])

        # Fallback: synthesize a linear growth curve if no equity series
        if strat is None or strat.size < 2:
            # Build a simple reference line from initial=1 to final=1+ret/100
            final = 1.0 + (summary.total_return / 100.0)
            xs = np.linspace(0, 100, 101)
            ys = np.linspace(1.0, final, 101)
            self._strat_curve.setData(xs, ys)

    @staticmethod
    def _extract_curve(blob: dict, keys: list[str]) -> Optional[np.ndarray]:
        for k in keys:
            v = blob.get(k)
            if v is None and isinstance(blob.get("metrics"), dict):
                v = blob["metrics"].get(k)
            if isinstance(v, list) and len(v) > 1:
                try:
                    arr = np.array(v, dtype=np.float64)
                    if np.isfinite(arr).all():
                        return arr
                except Exception:
                    continue
            if isinstance(v, dict):
                # Could be {"dates": [...], "values": [...]}
                vals = v.get("values") or v.get("equity")
                if isinstance(vals, list) and len(vals) > 1:
                    try:
                        return np.array(vals, dtype=np.float64)
                    except Exception:
                        pass
        return None
