"""Strategies workspace — strategy registry browser + on-demand backtest runner."""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    Qt, QAbstractTableModel, QDate, QModelIndex, QObject, QRunnable, QThreadPool,
    QTimer, Signal,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDateEdit, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSplitter, QTableView, QVBoxLayout,
    QWidget,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, BORDER_STRONG,
    TEXT, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT, ACCENT_SOFT, GAIN, LOSS, WARN,
    mono_font, ui_font,
)
from apps.desktop.utils import fmt_number, fmt_signed_pct
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.pane_header import PaneHeader
from apps.desktop.services.paper_service import PaperService
from apps.desktop.services.paper_types import StrategyEntry


# ---------------------------------------------------------------------------
# Strategy list model columns
# ---------------------------------------------------------------------------

_SCOLS = [
    ("name",   "NAME",   200, "left"),
    ("source", "SOURCE",  70, "left"),
    ("active", "●",       28, "center"),
]


# ---------------------------------------------------------------------------
# _StrategyModel
# ---------------------------------------------------------------------------

class _StrategyModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list[StrategyEntry] = []
        self._font      = ui_font(12)
        self._font_bold = ui_font(12, 600)
        self._font_mono = mono_font(12)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(_SCOLS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return _SCOLS[section][1]
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            a = _SCOLS[section][3]
            if a == "center":
                return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)
        return None

    def data(self, idx: QModelIndex, role: int = Qt.DisplayRole):
        if not idx.isValid():
            return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows):
            return None
        s = self._rows[r]
        key = _SCOLS[c][0]

        if role == Qt.DisplayRole:
            if key == "name":   return s.name
            if key == "source": return s.source.upper()
            if key == "active": return "●" if s.is_active else ""

        if role == Qt.TextAlignmentRole:
            a = _SCOLS[c][3]
            if a == "center":
                return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int((Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter)

        if role == Qt.ForegroundRole:
            if key == "name":   return QColor(TEXT)
            if key == "source":
                return QColor(ACCENT if s.source == "custom" else TEXT_MUTED)
            if key == "active":
                return QColor(GAIN if s.is_active else TEXT_MUTED)

        if role == Qt.FontRole:
            if key == "name":   return self._font_bold
            if key == "source": return self._font_mono
            if key == "active": return self._font

        if role == Qt.UserRole:
            return s

        return None

    def set_rows(self, rows: list[StrategyEntry]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row: int) -> Optional[StrategyEntry]:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None


# ---------------------------------------------------------------------------
# _StrategyDetail
# ---------------------------------------------------------------------------

def _make_sep() -> QFrame:
    sep = QFrame()
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background: {BORDER};")
    return sep


class _StrategyDetail(QFrame):
    """Right-top pane: shows name, description, phase-result KPIs, and config."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"_StrategyDetail {{ background: {PANE}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = PaneHeader("Strategy Detail")
        root.addWidget(self.header)

        # Scrollable body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        body = QFrame()
        body.setStyleSheet(f"QFrame {{ background: {PANE}; }}")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 14, 14, 14)
        bl.setSpacing(10)

        # Name label
        self._name_lbl = QLabel("—")
        self._name_lbl.setFont(ui_font(18, 700))
        self._name_lbl.setStyleSheet(f"color: {TEXT};")
        self._name_lbl.setWordWrap(True)
        bl.addWidget(self._name_lbl)

        # Description
        self._desc_lbl = QLabel("")
        self._desc_lbl.setFont(ui_font(12))
        self._desc_lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self._desc_lbl.setWordWrap(True)
        bl.addWidget(self._desc_lbl)

        bl.addWidget(_make_sep())

        # Phase results section header
        ph_hdr = QLabel("PHASE RESULTS")
        ph_hdr.setFont(ui_font(10, 600))
        ph_hdr.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1px;")
        bl.addWidget(ph_hdr)

        self._phase_grid = QGridLayout()
        self._phase_grid.setHorizontalSpacing(20)
        self._phase_grid.setVerticalSpacing(5)
        self._phase_grid.setColumnStretch(1, 1)
        bl.addLayout(self._phase_grid)

        self._no_phase_lbl = QLabel("No phase results available.")
        self._no_phase_lbl.setFont(mono_font(11))
        self._no_phase_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        bl.addWidget(self._no_phase_lbl)

        bl.addWidget(_make_sep())

        # Config section header
        cfg_hdr = QLabel("CONFIG")
        cfg_hdr.setFont(ui_font(10, 600))
        cfg_hdr.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1px;")
        bl.addWidget(cfg_hdr)

        self._cfg_grid = QGridLayout()
        self._cfg_grid.setHorizontalSpacing(20)
        self._cfg_grid.setVerticalSpacing(5)
        self._cfg_grid.setColumnStretch(1, 1)
        bl.addLayout(self._cfg_grid)

        self._no_cfg_lbl = QLabel("No config available.")
        self._no_cfg_lbl.setFont(mono_font(11))
        self._no_cfg_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        bl.addWidget(self._no_cfg_lbl)

        bl.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key_label(text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setFont(ui_font(10, 500))
        lbl.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 0.8px;")
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return lbl

    @staticmethod
    def _val_label(text: str, color: str = TEXT) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(mono_font(12, 500))
        lbl.setStyleSheet(f"color: {color};")
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return lbl

    def _clear_grid(self, grid: QGridLayout) -> None:
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_entry(self, s: Optional[StrategyEntry]) -> None:
        if s is None:
            self._name_lbl.setText("—")
            self._desc_lbl.setText("")
            self.header.set_subtitle("")
            self._clear_grid(self._phase_grid)
            self._clear_grid(self._cfg_grid)
            self._no_phase_lbl.show()
            self._no_cfg_lbl.show()
            return

        self._name_lbl.setText(s.name)
        self._desc_lbl.setText(s.description or "No description.")
        src_badge = " [custom]" if s.source == "custom" else " [builtin]"
        self.header.set_subtitle(s.id + src_badge)

        self._clear_grid(self._phase_grid)
        self._clear_grid(self._cfg_grid)

        # --- Phase results ---
        notes = s.notes or {}

        # Also check top-level notes keys directly
        _PHASE_KEYS = [
            ("full_7yr_return_pct",  "7yr Return",  True),
            ("oos_6yr_return_pct",   "OOS 6yr",     True),
            ("full_7yr_sharpe",      "Sharpe",      False),
            ("full_7yr_mdd_pct",     "MaxDD",       False),
            ("fwd_return_pct",       "FWD",         True),
            ("hist_return_pct",      "HIST",        True),
            ("score",                "Score",       False),
        ]

        row_i = 0
        phase_found = False
        for key, label, is_pct in _PHASE_KEYS:
            # Search in phase_data first, then notes directly
            val = phase_data.get(key)
            if val is None:
                val = notes.get(key)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue

            phase_found = True
            if is_pct:
                txt = fmt_signed_pct(fval)
                color = GAIN if fval >= 0 else LOSS
            elif key == "score":
                txt = f"{fval:.2f}"
                color = GAIN if fval >= 80 else (WARN if fval >= 60 else LOSS)
            else:
                txt = f"{fval:.3f}"
                color = GAIN if fval >= 1.0 else (WARN if fval >= 0.5 else LOSS)

            self._phase_grid.addWidget(self._key_label(label), row_i, 0)
            self._phase_grid.addWidget(self._val_label(txt, color), row_i, 1)
            row_i += 1

        if phase_found:
            self._no_phase_lbl.hide()
        else:
            self._no_phase_lbl.show()

        # --- Config ---
        cfg = s.config or {}
        _CFG_KEYS = [
            ("holding_days",        "Holding Days"),
            ("max_positions",       "Max Positions"),
            ("stop_loss_pct",       "Stop Loss %"),
            ("trailing_stop_pct",   "Trailing Stop %"),
            ("bear_threshold",      "Bear Threshold"),
        ]

        row_i = 0
        cfg_found = False
        for key, label in _CFG_KEYS:
            val = cfg.get(key)
            if val is None:
                continue
            try:
                txt = str(val)
            except Exception:
                continue
            cfg_found = True
            self._cfg_grid.addWidget(self._key_label(label), row_i, 0)
            self._cfg_grid.addWidget(self._val_label(txt), row_i, 1)
            row_i += 1

        if cfg_found:
            self._no_cfg_lbl.hide()
        else:
            self._no_cfg_lbl.show()


# ---------------------------------------------------------------------------
# Background worker for backtest
# ---------------------------------------------------------------------------

class _BtSignals(QObject):
    done  = Signal(dict)
    error = Signal(str)


class _BacktestWorker(QRunnable):
    def __init__(
        self,
        paper: PaperService,
        strategy_id: str,
        start_date: str,
        end_date: str,
        capital: float,
    ):
        super().__init__()
        self.paper       = paper
        self.strategy_id = strategy_id
        self.start_date  = start_date
        self.end_date    = end_date
        self.capital     = capital
        self.signals     = _BtSignals()

    def run(self) -> None:
        try:
            result = self.paper.run_backtest(
                self.strategy_id,
                self.start_date,
                self.end_date,
                self.capital,
            )
            self.signals.done.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# _RunKPI — tiny inline key/value pair for the metrics row
# ---------------------------------------------------------------------------

class _RunKPI(QFrame):
    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"_RunKPI {{ background: {ELEVATED}; border: 1px solid {BORDER}; }}")
        self.setFixedHeight(52)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(2)
        self._lbl = QLabel(label.upper())
        self._lbl.setFont(ui_font(10, 500))
        self._lbl.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 0.8px;")
        self._val = QLabel("—")
        self._val.setFont(mono_font(15, 500))
        self._val.setStyleSheet(f"color: {TEXT};")
        lay.addWidget(self._lbl)
        lay.addWidget(self._val)

    def set_value(self, text: str, color: Optional[str] = None) -> None:
        self._val.setText(text)
        self._val.setStyleSheet(f"color: {color or TEXT};")


# ---------------------------------------------------------------------------
# _BacktestRunner
# ---------------------------------------------------------------------------

class _BacktestRunner(QFrame):
    def __init__(self, paper: PaperService, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._paper       = paper
        self._strategy_id: Optional[str] = None
        self.setStyleSheet(f"_BacktestRunner {{ background: {BG}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Sub-header row -------------------------------------------------
        hdr = QFrame()
        hdr.setFixedHeight(32)
        hdr.setStyleSheet(
            f"QFrame {{ background: {PANE}; border-bottom: 1px solid {BORDER}; }}"
        )
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(10, 0, 10, 0)
        hdr_lay.setSpacing(12)

        title_lbl = QLabel("BACKTEST RUNNER")
        title_lbl.setFont(ui_font(11, 600))
        title_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 1.2px;")
        hdr_lay.addWidget(title_lbl)

        hdr_lay.addStretch(1)

        # Start date
        start_lbl = QLabel("Start")
        start_lbl.setFont(ui_font(11))
        start_lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        hdr_lay.addWidget(start_lbl)

        self._start_edit = QDateEdit()
        self._start_edit.setCalendarPopup(True)
        self._start_edit.setDate(QDate(2019, 1, 1))
        self._start_edit.setDisplayFormat("yyyy-MM-dd")
        self._start_edit.setFont(mono_font(11))
        self._start_edit.setFixedHeight(24)
        self._start_edit.setStyleSheet(
            f"QDateEdit {{ background: {ELEVATED}; color: {TEXT}; border: 1px solid {BORDER};"
            f"padding: 2px 6px; }}"
            f"QDateEdit:focus {{ border: 1px solid {ACCENT}; }}"
        )
        hdr_lay.addWidget(self._start_edit)

        end_lbl = QLabel("End")
        end_lbl.setFont(ui_font(11))
        end_lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        hdr_lay.addWidget(end_lbl)

        self._end_edit = QDateEdit()
        self._end_edit.setCalendarPopup(True)
        today = date.today()
        self._end_edit.setDate(QDate(today.year, today.month, today.day))
        self._end_edit.setDisplayFormat("yyyy-MM-dd")
        self._end_edit.setFont(mono_font(11))
        self._end_edit.setFixedHeight(24)
        self._end_edit.setStyleSheet(
            f"QDateEdit {{ background: {ELEVATED}; color: {TEXT}; border: 1px solid {BORDER};"
            f"padding: 2px 6px; }}"
            f"QDateEdit:focus {{ border: 1px solid {ACCENT}; }}"
        )
        hdr_lay.addWidget(self._end_edit)

        cap_lbl = QLabel("Capital")
        cap_lbl.setFont(ui_font(11))
        cap_lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        hdr_lay.addWidget(cap_lbl)

        self._capital_edit = QLineEdit("1000000")
        self._capital_edit.setFont(mono_font(11))
        self._capital_edit.setFixedHeight(24)
        self._capital_edit.setFixedWidth(110)
        self._capital_edit.setStyleSheet(
            f"QLineEdit {{ background: {ELEVATED}; color: {TEXT}; border: 1px solid {BORDER};"
            f"padding: 2px 6px; }}"
            f"QLineEdit:focus {{ border: 1px solid {ACCENT}; }}"
        )
        hdr_lay.addWidget(self._capital_edit)

        self._run_btn = QPushButton("▶ Run")
        self._run_btn.setFont(ui_font(11, 600))
        self._run_btn.setFixedHeight(24)
        self._run_btn.setStyleSheet(
            f"QPushButton {{ background: {ACCENT_SOFT}; color: {ACCENT};"
            f"border: 1px solid {ACCENT}; padding: 2px 12px; }}"
            f"QPushButton:hover {{ background: {ACCENT}; color: {BG}; }}"
            f"QPushButton:pressed {{ background: {ACCENT}; color: {BG}; }}"
            f"QPushButton:disabled {{ background: {ELEVATED}; color: {TEXT_MUTED};"
            f"border-color: {BORDER}; }}"
        )
        self._run_btn.clicked.connect(self._run)
        hdr_lay.addWidget(self._run_btn)

        root.addWidget(hdr)

        # ---- Plot -----------------------------------------------------------
        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.setBackground(BG)
        self._plot_widget.ci.setContentsMargins(0, 0, 0, 0)
        self._plot = self._plot_widget.addPlot(
            axisItems={
                "right":  pg.AxisItem("right"),
                "bottom": pg.AxisItem("bottom"),
            }
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

        self._strat_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QColor(ACCENT), width=1.6)
        )
        self._bench_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QColor(TEXT_MUTED), width=1.2, style=Qt.DashLine)
        )
        self._plot.addItem(self._strat_curve)
        self._plot.addItem(self._bench_curve)

        self._legend = pg.LegendItem(offset=(10, 10), labelTextColor=QColor(TEXT_SECONDARY))
        self._legend.setParentItem(self._plot.graphicsItem())
        self._legend.addItem(self._strat_curve, "Strategy")
        self._legend.addItem(self._bench_curve, "Benchmark")

        root.addWidget(self._plot_widget, 1)

        # ---- Metrics row ----------------------------------------------------
        metrics_frame = QFrame()
        metrics_frame.setFixedHeight(60)
        metrics_frame.setStyleSheet(
            f"QFrame {{ background: {PANE}; border-top: 1px solid {BORDER}; }}"
        )
        m_lay = QHBoxLayout(metrics_frame)
        m_lay.setContentsMargins(14, 4, 14, 4)
        m_lay.setSpacing(8)

        self._kpi_ret    = _RunKPI("Return")
        self._kpi_sharpe = _RunKPI("Sharpe")
        self._kpi_dd     = _RunKPI("Max DD")
        self._kpi_trades = _RunKPI("Trades")

        for kpi in (self._kpi_ret, self._kpi_sharpe, self._kpi_dd, self._kpi_trades):
            m_lay.addWidget(kpi)
        m_lay.addStretch(1)

        # Status label (run errors / running indicator)
        self._status_lbl = QLabel("")
        self._status_lbl.setFont(mono_font(11))
        self._status_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        m_lay.addWidget(self._status_lbl)

        root.addWidget(metrics_frame)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_strategy(self, strategy_id: Optional[str]) -> None:
        """Called by the parent Strategies widget when selection changes."""
        self._strategy_id = strategy_id

    def trigger_run(self) -> None:
        """Called externally by the header button."""
        self._run()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
                vals = v.get("values") or v.get("equity")
                if isinstance(vals, list) and len(vals) > 1:
                    try:
                        return np.array(vals, dtype=np.float64)
                    except Exception:
                        pass
        return None

    def _run(self) -> None:
        sid = self._strategy_id
        if not sid:
            self._status_lbl.setText("No strategy selected.")
            return

        start_str = self._start_edit.date().toString("yyyy-MM-dd")
        end_str   = self._end_edit.date().toString("yyyy-MM-dd")
        cap_text  = self._capital_edit.text().strip()

        try:
            capital = float(cap_text.replace(",", ""))
            if capital <= 0:
                raise ValueError("non-positive capital")
        except ValueError:
            self._status_lbl.setText("Invalid capital value.")
            return

        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")
        self._status_lbl.setText("")

        worker = _BacktestWorker(self._paper, sid, start_str, end_str, capital)
        worker.signals.done.connect(self._load_result)
        worker.signals.error.connect(self._on_error)
        QThreadPool.globalInstance().start(worker)

    def _load_result(self, result: dict) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶ Run")
        self._status_lbl.setText("")

        keys_strat = ["equity_curve", "strategy_equity", "equity", "portfolio_equity"]
        keys_bench = ["benchmark_equity", "nepse_equity", "benchmark", "index_equity"]

        strat = self._extract_curve(result, keys_strat)
        bench = self._extract_curve(result, keys_bench)

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

        # Metrics — try multiple key names
        def _pick(d: dict, *keys):
            for k in keys:
                v = d.get(k)
                if v is None and isinstance(d.get("metrics"), dict):
                    v = d["metrics"].get(k)
                if v is not None:
                    return v
            return None

        ret    = _pick(result, "total_return_pct", "total_return")
        sharpe = _pick(result, "sharpe_ratio", "sharpe")
        dd     = _pick(result, "max_drawdown_pct", "max_dd")
        trades = _pick(result, "trade_count", "n_trades")

        if ret is not None:
            try:
                fret = float(ret)
                self._kpi_ret.set_value(fmt_signed_pct(fret), GAIN if fret >= 0 else LOSS)
            except Exception:
                self._kpi_ret.set_value(str(ret))
        else:
            self._kpi_ret.set_value("—")

        if sharpe is not None:
            try:
                fsh = float(sharpe)
                self._kpi_sharpe.set_value(f"{fsh:.2f}", GAIN if fsh >= 1.0 else TEXT)
            except Exception:
                self._kpi_sharpe.set_value(str(sharpe))
        else:
            self._kpi_sharpe.set_value("—")

        if dd is not None:
            try:
                fdd = float(dd)
                self._kpi_dd.set_value(f"{fdd:.2f}%", LOSS)
            except Exception:
                self._kpi_dd.set_value(str(dd))
        else:
            self._kpi_dd.set_value("—")

        if trades is not None:
            self._kpi_trades.set_value(str(int(float(trades))))
        else:
            self._kpi_trades.set_value("—")

        # Synthesise equity curve from return if no equity series in result
        if strat is None or strat.size < 2:
            if ret is not None:
                try:
                    final = 1.0 + float(ret) / 100.0
                    xs = np.linspace(0, 100, 101)
                    ys = np.linspace(1.0, final, 101)
                    self._strat_curve.setData(xs, ys)
                except Exception:
                    pass

    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶ Run")
        self._status_lbl.setText(f"Error: {msg}")
        self._status_lbl.setStyleSheet(f"color: {LOSS};")


# ---------------------------------------------------------------------------
# Strategies  (main widget)
# ---------------------------------------------------------------------------

class Strategies(QWidget):
    """Strategies workspace — registry browser + on-demand backtest runner."""

    def __init__(
        self,
        paper: PaperService,
        link: LinkGroup,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._paper        = paper
        self._link         = link
        self._selected_id: Optional[str] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Header ----------------------------------------------------------
        self.header = PaneHeader("Strategies", link_color=link.color, subtitle="")

        # Set Active button
        self._btn_set_active = QPushButton("Set Active")
        self._btn_set_active.setFont(ui_font(11))
        self._btn_set_active.setFixedHeight(20)
        self._btn_set_active.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_SECONDARY};"
            f"border: 1px solid {BORDER_STRONG}; padding: 0 8px; }}"
            f"QPushButton:hover {{ color: {TEXT}; border-color: {TEXT_MUTED}; }}"
            f"QPushButton:pressed {{ background: {ELEVATED}; }}"
        )
        self._btn_set_active.clicked.connect(self._on_set_active)
        self.header.add_action(self._btn_set_active)

        # Run Backtest button
        self._btn_run_bt = QPushButton("▶ Run Backtest")
        self._btn_run_bt.setFont(ui_font(11, 600))
        self._btn_run_bt.setFixedHeight(20)
        self._btn_run_bt.setStyleSheet(
            f"QPushButton {{ background: {ACCENT_SOFT}; color: {ACCENT};"
            f"border: 1px solid {ACCENT}; padding: 0 8px; }}"
            f"QPushButton:hover {{ background: {ACCENT}; color: {BG}; }}"
            f"QPushButton:pressed {{ background: {ACCENT}; color: {BG}; }}"
        )
        self._btn_run_bt.clicked.connect(self._on_run_backtest)
        self.header.add_action(self._btn_run_bt)

        root.addWidget(self.header)

        # ---- Horizontal splitter (list | detail+runner) ----------------------
        h_split = QSplitter(Qt.Horizontal)
        h_split.setHandleWidth(1)

        # ---- LEFT: strategy list ----
        left = QFrame()
        left.setStyleSheet(f"QFrame {{ background: {PANE}; }}")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        list_hdr = PaneHeader("Strategy List")
        ll.addWidget(list_hdr)

        self._model = _StrategyModel()
        self.table = QTableView()
        self.table.setModel(self._model)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        for i, (_k, _h, w, _a) in enumerate(_SCOLS):
            self.table.setColumnWidth(i, w)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.clicked.connect(self._on_row_clicked)
        ll.addWidget(self.table, 1)

        h_split.addWidget(left)

        # ---- RIGHT: vertical splitter (detail | runner) ----
        right = QFrame()
        right.setStyleSheet(f"QFrame {{ background: {BG}; }}")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        v_split = QSplitter(Qt.Vertical)
        v_split.setHandleWidth(1)

        self._detail  = _StrategyDetail()
        self._runner  = _BacktestRunner(paper)

        v_split.addWidget(self._detail)
        v_split.addWidget(self._runner)
        v_split.setStretchFactor(0, 2)
        v_split.setStretchFactor(1, 3)
        v_split.setSizes([320, 480])

        rl.addWidget(v_split, 1)
        h_split.addWidget(right)

        h_split.setStretchFactor(0, 38)
        h_split.setStretchFactor(1, 62)
        h_split.setSizes([520, 880])

        root.addWidget(h_split, 1)

        # ---- Initial load (defer 50ms so layout is settled) ----
        QTimer.singleShot(50, self._refresh)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        rows = self._paper.list_strategies()
        self._model.set_rows(rows)
        self.header.set_subtitle(f"{len(rows)} strategies")

        if rows:
            # Select the active strategy by default, else first row
            active_row = 0
            for i, s in enumerate(rows):
                if s.is_active:
                    active_row = i
                    break
            self.table.selectRow(active_row)
            self._select_row(active_row)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_row_clicked(self, idx: QModelIndex) -> None:
        if not idx.isValid():
            return
        self._select_row(idx.row())

    def _select_row(self, row: int) -> None:
        entry = self._model.row_at(row)
        if entry is None:
            return
        self._selected_id = entry.id
        self._detail.set_entry(entry)
        self._runner.set_strategy(entry.id)

    def _on_set_active(self) -> None:
        if not self._selected_id:
            return
        try:
            self._paper.set_active_strategy(self._selected_id)
        except Exception:
            pass
        self._refresh()

    def _on_run_backtest(self) -> None:
        self._runner.trigger_run()
