"""Signals workspace — grid of recent signal fires + detail + linked chart preview.

Enhanced version:
  - STR and CONF columns (after SCORE)
  - Min-score filter spinbox in filter row
  - Regime banner (below header)
  - Signal type pills (below regime banner)
  - Data source: PaperService.generate_signals() via QRunnable worker
  - Keep existing ChartPane + detail card on right
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel,
    QTimer, QRunnable, QObject, Signal, QThreadPool,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QSplitter, QTableView, QVBoxLayout, QWidget,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, BORDER_STRONG,
    TEXT, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT, ACCENT_SOFT, GAIN, GAIN_HI, LOSS, WARN,
    mono_font, ui_font,
)
from apps.desktop.utils import fmt_signed_pct
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.chart_pane import ChartPane
from apps.desktop.widgets.pane_header import PaneHeader
from apps.desktop.services.paper_service import PaperService
from apps.desktop.services.paper_types import PaperSignal
from backend.core.services import MarketService


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_COLS = [
    ("symbol",      "SYMBOL",  88, "left"),
    ("signal_type", "TYPE",   130, "left"),
    ("score",       "SCORE",   72, "right"),
    ("strength",    "STR",     60, "right"),
    ("confidence",  "CONF",    60, "right"),
    ("regime",      "REGIME",  80, "left"),
    ("as_of",       "DATE",    96, "left"),
]

# Color palette for signal-type pills (cycling)
_PILL_COLORS = [ACCENT, GAIN, WARN, "#B087D6", "#3BC9DB"]


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _WorkerSignals(QObject):
    done = Signal(list, str)   # (signals: list[PaperSignal], regime: str)
    error = Signal(str)


class _SignalWorker(QRunnable):
    """Runs PaperService.generate_signals() off the UI thread."""

    def __init__(self, paper: PaperService):
        super().__init__()
        self.paper = paper
        self.signals = _WorkerSignals()

    def run(self):
        try:
            result = self.paper.generate_signals()
            # generate_signals() may return (list, str) or just list
            if isinstance(result, tuple) and len(result) == 2:
                sigs, regime = result
            else:
                sigs = result if result is not None else []
                regime = ""
            # Normalise regime to a string
            if regime is None:
                regime = ""
            self.signals.done.emit(list(sigs), str(regime))
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# _SignalsModel
# ---------------------------------------------------------------------------

class _SignalsModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list[PaperSignal] = []
        self._font = mono_font(12)
        self._font_bold = mono_font(12, 500)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return _COLS[section][1]
            if role == Qt.TextAlignmentRole:
                a = _COLS[section][3]
                return int(
                    (Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter
                )
        return None

    def data(self, idx: QModelIndex, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        r, c = idx.row(), idx.column()
        if r < 0 or r >= len(self._rows):
            return None
        s: PaperSignal = self._rows[r]
        key = _COLS[c][0]

        if role == Qt.DisplayRole:
            return self._display(s, key)

        if role == Qt.TextAlignmentRole:
            a = _COLS[c][3]
            return int(
                (Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter
            )

        if role == Qt.ForegroundRole:
            return QColor(self._fg_color(s, key))

        if role == Qt.FontRole:
            return self._font_bold if key == "symbol" else self._font

        if role == Qt.UserRole:
            return s

        return None

    def _display(self, s: PaperSignal, key: str) -> str:
        if key == "symbol":
            return getattr(s, "symbol", "") or "—"
        if key == "signal_type":
            return getattr(s, "signal_type", "") or "—"
        if key == "score":
            v = getattr(s, "score", None)
            return f"{v:.2f}" if v is not None else "—"
        if key == "strength":
            v = getattr(s, "strength", None)
            return f"{v:.2f}" if v is not None else "—"
        if key == "confidence":
            v = getattr(s, "confidence", None)
            return f"{v:.2f}" if v is not None else "—"
        if key == "regime":
            return getattr(s, "regime", None) or "—"
        if key == "as_of":
            val = getattr(s, "as_of", None)
            if val is None:
                return "—"
            if hasattr(val, "isoformat"):
                return val.isoformat()
            return str(val)
        return "—"

    def _fg_color(self, s: PaperSignal, key: str) -> str:
        if key == "symbol":
            return TEXT
        if key == "score":
            v = getattr(s, "score", 0.0) or 0.0
            return GAIN if v >= 0 else LOSS
        if key == "regime":
            reg = (getattr(s, "regime", "") or "").lower()
            if reg == "bull":
                return GAIN
            if reg == "bear":
                return LOSS
            if reg == "neutral":
                return TEXT_SECONDARY
            return TEXT_SECONDARY
        if key in ("strength", "confidence"):
            return TEXT_SECONDARY
        return TEXT_SECONDARY

    def set_rows(self, rows: list[PaperSignal]):
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, r: int) -> Optional[PaperSignal]:
        if 0 <= r < len(self._rows):
            return self._rows[r]
        return None


# ---------------------------------------------------------------------------
# _Proxy — text filter + min-score filter
# ---------------------------------------------------------------------------

class _Proxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self._min_score: float = 0.0

    def set_min_score(self, v: float):
        self._min_score = v
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        m = self.sourceModel()
        sig: Optional[PaperSignal] = m.data(m.index(source_row, 0, source_parent), Qt.UserRole)

        # Min-score gate
        if sig is not None:
            score = getattr(sig, "score", 0.0) or 0.0
            if score < self._min_score:
                return False

        # Text pattern gate
        pattern = self.filterRegularExpression().pattern().upper()
        if not pattern:
            return True

        # Check symbol, signal_type, regime columns (0, 1, 5)
        for col in (0, 1, 5):
            idx = m.index(source_row, col, source_parent)
            val = str(m.data(idx, Qt.DisplayRole) or "").upper()
            if pattern in val:
                return True

        return False


# ---------------------------------------------------------------------------
# _DetailCard — right column, selected signal summary
# ---------------------------------------------------------------------------

def _detail_sep() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER};")
    return f


class _DetailCard(QFrame):
    """Summarises the currently selected PaperSignal."""

    def __init__(self, link: LinkGroup, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._link = link
        self.setStyleSheet(f"_DetailCard {{ background: {PANE}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = PaneHeader("Signal Detail", link_color=link.color)
        root.addWidget(self.header)

        body = QFrame()
        body.setStyleSheet(f"QFrame {{ background: {PANE}; }}")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 12, 14, 12)
        bl.setSpacing(10)

        self._sym = QLabel("—")
        self._sym.setFont(ui_font(22, 600))
        self._sym.setStyleSheet(f"color: {TEXT};")
        bl.addWidget(self._sym)

        self._type_lbl = QLabel("")
        self._type_lbl.setFont(mono_font(13, 500))
        self._type_lbl.setStyleSheet(f"color: {ACCENT};")
        bl.addWidget(self._type_lbl)

        bl.addWidget(_detail_sep())

        # KV grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)

        def kv(label: str, row: int) -> QLabel:
            kl = QLabel(label.upper())
            kl.setFont(ui_font(10, 500))
            kl.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 0.8px;")
            kl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            vl = QLabel("—")
            vl.setFont(mono_font(12, 500))
            vl.setStyleSheet(f"color: {TEXT};")
            grid.addWidget(kl, row, 0)
            grid.addWidget(vl, row, 1)
            return vl

        self._v_score      = kv("Score",      0)
        self._v_strength   = kv("Strength",   1)
        self._v_confidence = kv("Confidence", 2)
        self._v_regime     = kv("Regime",     3)
        self._v_date       = kv("As Of",      4)
        bl.addLayout(grid)

        bl.addStretch(1)
        root.addWidget(body, 1)

        link.symbol_changed.connect(lambda s: self._sym.setText(s or "—"))

    def set_signal(self, s: Optional[PaperSignal]):
        if s is None:
            self._sym.setText("—")
            self._type_lbl.setText("")
            for v in (
                self._v_score, self._v_strength, self._v_confidence,
                self._v_regime, self._v_date,
            ):
                v.setText("—")
                v.setStyleSheet(f"color: {TEXT};")
            return

        self._sym.setText(getattr(s, "symbol", "—") or "—")
        self._type_lbl.setText(getattr(s, "signal_type", "") or "")

        score = getattr(s, "score", None)
        if score is not None:
            self._v_score.setText(f"{score:.2f}")
            self._v_score.setStyleSheet(f"color: {GAIN if score >= 0 else LOSS};")
        else:
            self._v_score.setText("—")
            self._v_score.setStyleSheet(f"color: {TEXT};")

        strength = getattr(s, "strength", None)
        self._v_strength.setText(f"{strength:.2f}" if strength is not None else "—")
        self._v_strength.setStyleSheet(f"color: {TEXT_SECONDARY};")

        conf = getattr(s, "confidence", None)
        self._v_confidence.setText(f"{conf:.2f}" if conf is not None else "—")
        self._v_confidence.setStyleSheet(f"color: {TEXT_SECONDARY};")

        regime = getattr(s, "regime", None) or "—"
        self._v_regime.setText(regime)
        reg = regime.lower()
        reg_color = GAIN if reg == "bull" else LOSS if reg == "bear" else TEXT
        self._v_regime.setStyleSheet(f"color: {reg_color};")

        as_of = getattr(s, "as_of", None)
        if as_of is not None:
            date_str = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
        else:
            date_str = "—"
        self._v_date.setText(date_str)
        self._v_date.setStyleSheet(f"color: {TEXT_SECONDARY};")


# ---------------------------------------------------------------------------
# Signals — main workspace
# ---------------------------------------------------------------------------

class Signals(QWidget):
    def __init__(
        self,
        paper: PaperService,
        market: MarketService,
        link: LinkGroup,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._paper = paper
        self._market = market
        self._link = link
        self._pill_widgets: list[QLabel] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        h_split = QSplitter(Qt.Horizontal)
        h_split.setHandleWidth(1)

        # ================================================================
        # LEFT PANE
        # ================================================================
        left = QFrame()
        left.setStyleSheet(f"QFrame {{ background: {PANE}; }}")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        # ---- PaneHeader ------------------------------------------------
        self.header = PaneHeader("Signals", subtitle="")
        ll.addWidget(self.header)

        # ---- Regime banner (hidden until data loaded) ------------------
        self._regime_banner = QFrame()
        self._regime_banner.setFixedHeight(28)
        self._regime_banner.setStyleSheet(
            f"QFrame {{ background: {ELEVATED}; border-bottom: 1px solid {BORDER}; }}"
        )
        rbl = QHBoxLayout(self._regime_banner)
        rbl.setContentsMargins(12, 0, 12, 0)
        rbl.setSpacing(14)

        regime_key = QLabel("REGIME:")
        regime_key.setFont(ui_font(10, 600))
        regime_key.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 0.8px;")
        rbl.addWidget(regime_key)

        self._regime_val = QLabel("—")
        self._regime_val.setFont(mono_font(12, 600))
        self._regime_val.setStyleSheet(f"color: {TEXT};")
        rbl.addWidget(self._regime_val)

        rbl.addWidget(self._dot_sep("●"))

        adv_key = QLabel("Advancers")
        adv_key.setFont(ui_font(10))
        adv_key.setStyleSheet(f"color: {TEXT_MUTED};")
        rbl.addWidget(adv_key)

        self._adv_val = QLabel("—")
        self._adv_val.setFont(mono_font(11, 500))
        self._adv_val.setStyleSheet(f"color: {GAIN};")
        rbl.addWidget(self._adv_val)

        rbl.addWidget(self._dot_sep("·"))

        dec_key = QLabel("Decliners")
        dec_key.setFont(ui_font(10))
        dec_key.setStyleSheet(f"color: {TEXT_MUTED};")
        rbl.addWidget(dec_key)

        self._dec_val = QLabel("—")
        self._dec_val.setFont(mono_font(11, 500))
        self._dec_val.setStyleSheet(f"color: {LOSS};")
        rbl.addWidget(self._dec_val)

        rbl.addWidget(self._dot_sep("·"))

        unch_key = QLabel("Unchanged")
        unch_key.setFont(ui_font(10))
        unch_key.setStyleSheet(f"color: {TEXT_MUTED};")
        rbl.addWidget(unch_key)

        self._unch_val = QLabel("—")
        self._unch_val.setFont(mono_font(11, 500))
        self._unch_val.setStyleSheet(f"color: {TEXT_SECONDARY};")
        rbl.addWidget(self._unch_val)

        rbl.addStretch(1)
        self._regime_banner.hide()
        ll.addWidget(self._regime_banner)

        # ---- Signal type pills row ------------------------------------
        self._pills_row = QFrame()
        self._pills_row.setFixedHeight(28)
        self._pills_row.setStyleSheet(
            f"QFrame {{ background: {PANE}; border-bottom: 1px solid {BORDER}; }}"
        )
        self._pills_layout = QHBoxLayout(self._pills_row)
        self._pills_layout.setContentsMargins(10, 4, 10, 4)
        self._pills_layout.setSpacing(6)
        self._pills_row.hide()
        ll.addWidget(self._pills_row)

        # ---- Filter row ------------------------------------------------
        filt = QFrame()
        filt.setStyleSheet(
            f"QFrame {{ background: {PANE}; border-bottom: 1px solid {BORDER}; }}"
        )
        filt.setFixedHeight(32)
        fl = QHBoxLayout(filt)
        fl.setContentsMargins(10, 4, 10, 4)
        fl.setSpacing(8)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("🔍  filter by symbol, type, regime")
        self.filter_edit.setFont(mono_font(12))
        self.filter_edit.setFixedHeight(22)
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._on_filter_text)
        fl.addWidget(self.filter_edit, 1)

        min_score_lbl = QLabel("Min score:")
        min_score_lbl.setFont(ui_font(11))
        min_score_lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        fl.addWidget(min_score_lbl)

        self._min_score_spin = QDoubleSpinBox()
        self._min_score_spin.setMinimum(0.0)
        self._min_score_spin.setMaximum(99.0)
        self._min_score_spin.setSingleStep(0.1)
        self._min_score_spin.setDecimals(2)
        self._min_score_spin.setValue(0.0)
        self._min_score_spin.setFixedWidth(72)
        self._min_score_spin.setFixedHeight(22)
        self._min_score_spin.setFont(mono_font(12))
        self._min_score_spin.setStyleSheet(
            f"QDoubleSpinBox {{"
            f"  background: {BG};"
            f"  color: {TEXT};"
            f"  border: 1px solid {BORDER};"
            f"  padding: 1px 4px;"
            f"}}"
            f"QDoubleSpinBox:focus {{"
            f"  border: 1px solid {ACCENT};"
            f"}}"
            f"QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{"
            f"  background: {BORDER};"
            f"  width: 14px;"
            f"}}"
        )
        self._min_score_spin.valueChanged.connect(self._on_min_score_changed)
        fl.addWidget(self._min_score_spin)

        self._count = QLabel("0 signals")
        self._count.setFont(mono_font(11, 500))
        self._count.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self._count.setMinimumWidth(72)
        fl.addWidget(self._count)

        ll.addWidget(filt)

        # ---- Signals table --------------------------------------------
        self._model = _SignalsModel()
        self._proxy = _Proxy()
        self._proxy.setSourceModel(self._model)

        self.table = QTableView()
        self.table.setModel(self._proxy)
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        for i, (_k, _h, w, _a) in enumerate(_COLS):
            self.table.setColumnWidth(i, w)
        self.table.clicked.connect(self._on_row_clicked)
        ll.addWidget(self.table, 1)

        h_split.addWidget(left)

        # ================================================================
        # RIGHT PANE
        # ================================================================
        right = QFrame()
        right.setStyleSheet(f"QFrame {{ background: {BG}; }}")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        v_split = QSplitter(Qt.Vertical)
        v_split.setHandleWidth(1)

        self.detail = _DetailCard(link)
        v_split.addWidget(self.detail)

        self.chart = ChartPane(market, link, default_window_bars=132)
        v_split.addWidget(self.chart)

        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 3)
        v_split.setSizes([240, 560])
        rl.addWidget(v_split)

        h_split.addWidget(right)
        h_split.setStretchFactor(0, 3)
        h_split.setStretchFactor(1, 4)
        h_split.setSizes([620, 820])
        root.addWidget(h_split)

        # ---- Timers / initial load ------------------------------------
        # 300s auto-refresh (signal generation is expensive)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(300_000)

        QTimer.singleShot(200, self._refresh)

    # ------------------------------------------------------------------
    # Static helper: small separator dot label
    # ------------------------------------------------------------------

    @staticmethod
    def _dot_sep(char: str = "●") -> QLabel:
        lbl = QLabel(char)
        lbl.setFont(ui_font(8))
        lbl.setStyleSheet(f"color: {BORDER_STRONG};")
        return lbl

    # ------------------------------------------------------------------
    # Refresh — spawns worker
    # ------------------------------------------------------------------

    def _refresh(self):
        self.header.set_subtitle("Loading...")
        worker = _SignalWorker(self._paper)
        worker.signals.done.connect(self._on_worker_done)
        worker.signals.error.connect(self._on_worker_error)
        QThreadPool.globalInstance().start(worker)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_worker_done(self, sigs: list[PaperSignal], regime: str):
        self._model.set_rows(sigs)
        self._update_regime_banner(sigs, regime)
        self._update_pills(sigs)
        self._update_count()

        if sigs:
            self.table.selectRow(0)
            self._on_row_clicked(self._proxy.index(0, 0))

    def _on_worker_error(self, msg: str):
        self.header.set_subtitle(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Regime banner update
    # ------------------------------------------------------------------

    def _update_regime_banner(self, sigs: list[PaperSignal], regime: str):
        reg = regime.strip().lower() if regime else ""

        if reg == "bull":
            reg_color = GAIN
            reg_text = "BULL"
        elif reg == "bear":
            reg_color = LOSS
            reg_text = "BEAR"
        elif reg == "neutral":
            reg_color = TEXT_SECONDARY
            reg_text = "NEUTRAL"
        else:
            reg_color = TEXT_SECONDARY
            reg_text = regime.upper() if regime else "—"

        self._regime_val.setText(reg_text)
        self._regime_val.setStyleSheet(f"color: {reg_color}; font-weight: 600;")

        # Compute advancers / decliners / unchanged using signal scores
        adv = sum(1 for s in sigs if (getattr(s, "score", 0) or 0) > 0)
        dec = sum(1 for s in sigs if (getattr(s, "score", 0) or 0) < 0)
        unch = len(sigs) - adv - dec
        self._adv_val.setText(str(adv))
        self._dec_val.setText(str(dec))
        self._unch_val.setText(str(unch))

        if sigs:
            self._regime_banner.show()
        else:
            self._regime_banner.hide()

    # ------------------------------------------------------------------
    # Signal type pills update
    # ------------------------------------------------------------------

    def _update_pills(self, sigs: list[PaperSignal]):
        # Remove old pill widgets
        for w in self._pill_widgets:
            self._pills_layout.removeWidget(w)
            w.deleteLater()
        self._pill_widgets.clear()

        # Remove any remaining stretch items
        while self._pills_layout.count() > 0:
            item = self._pills_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not sigs:
            self._pills_row.hide()
            return

        # Count by signal type
        counts: Counter = Counter()
        for s in sigs:
            st = getattr(s, "signal_type", None) or "unknown"
            counts[st] += 1

        for i, (sig_type, count) in enumerate(counts.most_common()):
            color = _PILL_COLORS[i % len(_PILL_COLORS)]
            pill = QLabel(f"{sig_type}: {count}")
            pill.setFont(ui_font(10, 500))
            pill.setStyleSheet(
                f"QLabel {{"
                f"  color: {color};"
                f"  border: 1px solid {color};"
                f"  padding: 1px 6px;"
                f"  background: transparent;"
                f"}}"
            )
            pill.setFixedHeight(18)
            self._pills_layout.addWidget(pill)
            self._pill_widgets.append(pill)

        self._pills_layout.addStretch(1)
        self._pills_row.show()

    # ------------------------------------------------------------------
    # Count update
    # ------------------------------------------------------------------

    def _update_count(self):
        visible = self._proxy.rowCount()
        total = self._model.rowCount()
        if visible == total:
            self._count.setText(f"{total} signals")
            self.header.set_subtitle(f"{total} signals")
        else:
            self._count.setText(f"{visible} / {total}")
            self.header.set_subtitle(f"{visible} / {total} signals")

    # ------------------------------------------------------------------
    # Filter handlers
    # ------------------------------------------------------------------

    def _on_filter_text(self, text: str):
        self._proxy.setFilterFixedString(text.strip())
        self._update_count()

    def _on_min_score_changed(self, value: float):
        self._proxy.set_min_score(value)
        self._update_count()

    # ------------------------------------------------------------------
    # Row click
    # ------------------------------------------------------------------

    def _on_row_clicked(self, idx: QModelIndex):
        if not idx.isValid():
            return
        src_row = self._proxy.mapToSource(idx).row()
        s = self._model.row_at(src_row)
        if s is None:
            return
        self.detail.set_signal(s)
        sym = getattr(s, "symbol", None)
        if sym:
            self._link.set_symbol(sym)
