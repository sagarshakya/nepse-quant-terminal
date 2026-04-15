"""Orders workspace — pending/filled paper order book + trade execution."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel,
    QTimer, QRunnable, QObject, Signal, QThreadPool,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSplitter, QTabWidget, QTableView, QVBoxLayout, QWidget,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, BORDER_STRONG,
    TEXT, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT, ACCENT_SOFT, GAIN, LOSS, WARN,
    mono_font, ui_font,
)
from apps.desktop.utils import fmt_number
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.pane_header import PaneHeader
from apps.desktop.widgets.trade_dialog import TradeDialog
from apps.desktop.services.paper_service import PaperService
from apps.desktop.services.paper_types import PaperOrder


# ---------------------------------------------------------------------------
# Button stylesheets
# ---------------------------------------------------------------------------

_BTN_BUY = (
    f"QPushButton {{"
    f"  background: transparent;"
    f"  color: {GAIN};"
    f"  border: 1px solid {GAIN};"
    f"  padding: 2px 10px;"
    f"  font-size: 11px;"
    f"}}"
    f"QPushButton:hover {{"
    f"  background: {GAIN};"
    f"  color: {BG};"
    f"}}"
)

_BTN_SELL = (
    f"QPushButton {{"
    f"  background: transparent;"
    f"  color: {LOSS};"
    f"  border: 1px solid {LOSS};"
    f"  padding: 2px 10px;"
    f"  font-size: 11px;"
    f"}}"
    f"QPushButton:hover {{"
    f"  background: {LOSS};"
    f"  color: {BG};"
    f"}}"
)

_BTN_MATCH = (
    f"QPushButton {{"
    f"  background: transparent;"
    f"  color: {ACCENT};"
    f"  border: 1px solid {ACCENT};"
    f"  padding: 2px 10px;"
    f"  font-size: 11px;"
    f"}}"
    f"QPushButton:hover {{"
    f"  background: {ACCENT};"
    f"  color: {BG};"
    f"}}"
)


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_COLS = [
    ("status",       "STATUS",       90,  "left"),
    ("symbol",       "SYMBOL",       88,  "left"),
    ("action",       "ACTION",       68,  "left"),
    ("qty",          "QTY",          72,  "right"),
    ("order_price",  "ORDER PRICE",  100, "right"),
    ("fill_price",   "FILL PRICE",   100, "right"),
    ("placed_at",    "PLACED AT",    130, "left"),
    ("reason",       "REASON",       160, "left"),
]


# ---------------------------------------------------------------------------
# Background worker: match orders
# ---------------------------------------------------------------------------

class _MatchWorkerSignals(QObject):
    done = Signal(int)   # number of orders matched
    error = Signal(str)


class _MatchWorker(QRunnable):
    def __init__(self, paper: PaperService):
        super().__init__()
        self.paper = paper
        self.signals = _MatchWorkerSignals()

    def run(self):
        try:
            result = self.paper.match_orders()
            # result may be int (count) or an object with a count attribute
            if isinstance(result, int):
                count = result
            elif hasattr(result, "matched"):
                count = result.matched
            else:
                count = 0
            self.signals.done.emit(count)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# _OrdersModel — table model for lists of PaperOrder
# ---------------------------------------------------------------------------

class _OrdersModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._rows: list[PaperOrder] = []
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
        order: PaperOrder = self._rows[r]
        key = _COLS[c][0]

        if role == Qt.DisplayRole:
            return self._display(order, key)

        if role == Qt.TextAlignmentRole:
            a = _COLS[c][3]
            return int(
                (Qt.AlignLeft if a == "left" else Qt.AlignRight) | Qt.AlignVCenter
            )

        if role == Qt.ForegroundRole:
            return QColor(self._fg_color(order, key))

        if role == Qt.FontRole:
            return self._font_bold if key in ("symbol", "status") else self._font

        if role == Qt.UserRole:
            return order

        return None

    def _display(self, order: PaperOrder, key: str) -> str:
        if key == "status":
            st = (getattr(order, "status", "") or "").upper()
            if st == "OPEN":
                return "● OPEN"
            if st == "FILLED":
                return "✓ FILLED"
            if st == "CANCELLED":
                return "✗ CANCELLED"
            return st

        if key == "symbol":
            return getattr(order, "symbol", "") or "—"

        if key == "action":
            return (getattr(order, "action", "") or "").upper()

        if key == "qty":
            qty = getattr(order, "qty", None)
            if qty is None:
                qty = getattr(order, "quantity", None)
            return f"{int(qty):,}" if qty is not None else "—"

        if key == "order_price":
            p = getattr(order, "order_price", None)
            if p is None:
                p = getattr(order, "price", None)
            return fmt_number(float(p)) if p is not None else "—"

        if key == "fill_price":
            p = getattr(order, "fill_price", None)
            return fmt_number(float(p)) if p is not None else "—"

        if key == "placed_at":
            ts = getattr(order, "created_at", None)
            if ts is None:
                ts = getattr(order, "placed_at", None)
            if ts is None:
                return "—"
            return str(ts)[:16]

        if key == "reason":
            return getattr(order, "reason", None) or "—"

        return "—"

    def _fg_color(self, order: PaperOrder, key: str) -> str:
        if key == "status":
            st = (getattr(order, "status", "") or "").upper()
            if st == "OPEN":
                return ACCENT
            if st == "FILLED":
                return GAIN
            if st == "CANCELLED":
                return TEXT_MUTED
            return TEXT_SECONDARY

        if key == "action":
            action = (getattr(order, "action", "") or "").upper()
            return GAIN if action == "BUY" else LOSS

        if key == "symbol":
            return TEXT

        if key in ("order_price", "fill_price", "qty"):
            return TEXT

        return TEXT_SECONDARY

    def set_rows(self, rows: list[PaperOrder]):
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, r: int) -> Optional[PaperOrder]:
        if 0 <= r < len(self._rows):
            return self._rows[r]
        return None


# ---------------------------------------------------------------------------
# _OrderDetailPanel — right pane showing selected order's full details
# ---------------------------------------------------------------------------

def _detail_sep() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER};")
    return f


class _OrderDetailPanel(QFrame):
    cancel_requested = Signal(object)  # emits PaperOrder

    def __init__(self, paper: PaperService, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._paper = paper
        self._current_order: Optional[PaperOrder] = None
        self.setStyleSheet(f"_OrderDetailPanel {{ background: {PANE}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = PaneHeader("Order Detail")
        root.addWidget(self.header)

        body = QFrame()
        body.setStyleSheet(f"QFrame {{ background: {PANE}; }}")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(16, 14, 16, 14)
        bl.setSpacing(10)

        # Symbol — large
        self._sym_lbl = QLabel("—")
        self._sym_lbl.setFont(ui_font(22, 700))
        self._sym_lbl.setStyleSheet(f"color: {TEXT};")
        bl.addWidget(self._sym_lbl)

        # Action badge
        self._action_lbl = QLabel("")
        self._action_lbl.setFont(mono_font(13, 600))
        self._action_lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        bl.addWidget(self._action_lbl)

        bl.addWidget(_detail_sep())

        # KV grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 1)

        def _kv(label: str, row: int) -> QLabel:
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

        self._v_status     = _kv("Status",      0)
        self._v_qty        = _kv("Quantity",     1)
        self._v_ord_price  = _kv("Order Price",  2)
        self._v_fill_price = _kv("Fill Price",   3)
        self._v_slippage   = _kv("Slippage",     4)
        self._v_placed     = _kv("Placed At",    5)
        self._v_updated    = _kv("Updated At",   6)
        self._v_reason     = _kv("Reason",       7)

        bl.addLayout(grid)
        bl.addWidget(_detail_sep())

        # Cancel button — only shown for OPEN orders
        self._cancel_btn = QPushButton("Cancel Order")
        self._cancel_btn.setFont(ui_font(12))
        self._cancel_btn.setFixedHeight(28)
        self._cancel_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent;"
            f"  color: {LOSS};"
            f"  border: 1px solid {LOSS};"
            f"  padding: 2px 12px;"
            f"  font-size: 11px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: {LOSS};"
            f"  color: {BG};"
            f"}}"
        )
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._cancel_btn.hide()
        bl.addWidget(self._cancel_btn, 0, Qt.AlignLeft)

        bl.addStretch(1)
        root.addWidget(body, 1)

    def set_order(self, order: Optional[PaperOrder]):
        self._current_order = order
        if order is None:
            self._sym_lbl.setText("—")
            self._action_lbl.setText("")
            for v in (
                self._v_status, self._v_qty, self._v_ord_price,
                self._v_fill_price, self._v_slippage, self._v_placed,
                self._v_updated, self._v_reason,
            ):
                v.setText("—")
                v.setStyleSheet(f"color: {TEXT};")
            self._cancel_btn.hide()
            return

        symbol = getattr(order, "symbol", "") or "—"
        action = (getattr(order, "action", "") or "").upper()
        status = (getattr(order, "status", "") or "").upper()

        self._sym_lbl.setText(symbol)

        action_color = GAIN if action == "BUY" else LOSS
        self._action_lbl.setText(action)
        self._action_lbl.setStyleSheet(f"color: {action_color};")

        # Status
        if status == "OPEN":
            st_text = "● OPEN"
            st_color = ACCENT
        elif status == "FILLED":
            st_text = "✓ FILLED"
            st_color = GAIN
        elif status == "CANCELLED":
            st_text = "✗ CANCELLED"
            st_color = TEXT_MUTED
        else:
            st_text = status
            st_color = TEXT_SECONDARY
        self._v_status.setText(st_text)
        self._v_status.setStyleSheet(f"color: {st_color};")

        # Quantity
        qty = getattr(order, "qty", None)
        if qty is None:
            qty = getattr(order, "quantity", None)
        self._v_qty.setText(f"{int(qty):,}" if qty is not None else "—")
        self._v_qty.setStyleSheet(f"color: {TEXT};")

        # Order price
        op = getattr(order, "order_price", None)
        if op is None:
            op = getattr(order, "price", None)
        self._v_ord_price.setText(f"NPR {fmt_number(float(op))}" if op is not None else "—")
        self._v_ord_price.setStyleSheet(f"color: {TEXT};")

        # Fill price
        fp = getattr(order, "fill_price", None)
        self._v_fill_price.setText(f"NPR {fmt_number(float(fp))}" if fp is not None else "—")
        self._v_fill_price.setStyleSheet(f"color: {GAIN if fp is not None else TEXT_MUTED};")

        # Slippage
        slip = getattr(order, "slippage", None)
        self._v_slippage.setText(f"{slip:.2f}%" if slip is not None else "—")
        self._v_slippage.setStyleSheet(f"color: {TEXT_SECONDARY};")

        # Timestamps
        created = getattr(order, "created_at", None) or getattr(order, "placed_at", None)
        updated = getattr(order, "updated_at", None)
        self._v_placed.setText(str(created)[:19] if created else "—")
        self._v_placed.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self._v_updated.setText(str(updated)[:19] if updated else "—")
        self._v_updated.setStyleSheet(f"color: {TEXT_SECONDARY};")

        # Reason
        reason = getattr(order, "reason", None) or "—"
        self._v_reason.setText(reason)
        self._v_reason.setStyleSheet(f"color: {TEXT_SECONDARY};")

        # Cancel button visibility
        if status == "OPEN":
            self._cancel_btn.show()
        else:
            self._cancel_btn.hide()

    def _on_cancel_clicked(self):
        if self._current_order is not None:
            self.cancel_requested.emit(self._current_order)


# ---------------------------------------------------------------------------
# Helper: build a configured QTableView for orders
# ---------------------------------------------------------------------------

def _make_table(model: _OrdersModel) -> QTableView:
    table = QTableView()
    table.setModel(model)
    table.setSortingEnabled(True)
    table.setShowGrid(False)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(24)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.horizontalHeader().setHighlightSections(False)
    table.setSelectionBehavior(QTableView.SelectRows)
    table.setEditTriggers(QTableView.NoEditTriggers)
    table.setAlternatingRowColors(False)
    for i, (_k, _h, w, _a) in enumerate(_COLS):
        table.setColumnWidth(i, w)
    return table


# ---------------------------------------------------------------------------
# Orders — main workspace widget
# ---------------------------------------------------------------------------

class Orders(QWidget):
    """Paper order management workspace with BUY/SELL/Match actions."""

    def __init__(
        self,
        paper: PaperService,
        link: LinkGroup,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._paper = paper
        self._link = link
        self._all_orders: list[PaperOrder] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- PaneHeader with action buttons --------------------------
        self.header = PaneHeader("Orders", link_color=link.color, subtitle="")

        self._btn_buy = QPushButton("BUY ▶")
        self._btn_buy.setFont(ui_font(11, 600))
        self._btn_buy.setFixedHeight(22)
        self._btn_buy.setStyleSheet(_BTN_BUY)

        self._btn_sell = QPushButton("SELL ▶")
        self._btn_sell.setFont(ui_font(11, 600))
        self._btn_sell.setFixedHeight(22)
        self._btn_sell.setStyleSheet(_BTN_SELL)

        self._btn_match = QPushButton("Match ▶")
        self._btn_match.setFont(ui_font(11, 600))
        self._btn_match.setFixedHeight(22)
        self._btn_match.setStyleSheet(_BTN_MATCH)

        self.header.add_action(self._btn_buy)
        self.header.add_action(self._btn_sell)
        self.header.add_action(self._btn_match)
        root.addWidget(self.header)

        # ---- Splitter: left (tabs) + right (detail) ------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # -- LEFT: QTabWidget with three tabs --------------------------
        left = QFrame()
        left.setStyleSheet(f"QFrame {{ background: {PANE}; }}")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            f"QTabWidget::pane {{"
            f"  border: none;"
            f"  background: {PANE};"
            f"}}"
            f"QTabBar::tab {{"
            f"  background: {PANE};"
            f"  color: {TEXT_SECONDARY};"
            f"  border: none;"
            f"  border-bottom: 2px solid transparent;"
            f"  padding: 4px 14px;"
            f"  font-size: 11px;"
            f"}}"
            f"QTabBar::tab:selected {{"
            f"  color: {TEXT};"
            f"  border-bottom: 2px solid {ACCENT};"
            f"}}"
            f"QTabBar::tab:hover {{"
            f"  color: {TEXT};"
            f"}}"
        )

        # Open Orders tab
        self._model_open = _OrdersModel()
        self._table_open = _make_table(self._model_open)
        self._table_open.clicked.connect(self._on_open_row_clicked)
        self._tabs.addTab(self._table_open, "Open Orders")

        # All Orders tab
        self._model_all = _OrdersModel()
        self._table_all = _make_table(self._model_all)
        self._table_all.clicked.connect(self._on_all_row_clicked)
        self._tabs.addTab(self._table_all, "All Orders")

        # Today's Trades tab (filled orders only)
        self._model_today = _OrdersModel()
        self._table_today = _make_table(self._model_today)
        self._table_today.clicked.connect(self._on_today_row_clicked)
        self._tabs.addTab(self._table_today, "Today's Trades")

        ll.addWidget(self._tabs)
        splitter.addWidget(left)

        # -- RIGHT: detail panel ---------------------------------------
        self._detail = _OrderDetailPanel(paper)
        self._detail.cancel_requested.connect(self._on_cancel_order)
        splitter.addWidget(self._detail)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([600, 400])
        root.addWidget(splitter, 1)

        # ---- Button connections --------------------------------------
        self._btn_buy.clicked.connect(self._open_buy_dialog)
        self._btn_sell.clicked.connect(self._open_sell_dialog)
        self._btn_match.clicked.connect(self._run_match)

        # ---- Link context changes ------------------------------------
        link.changed.connect(lambda ctx: None)  # keep reference alive

        # ---- Auto-refresh timer (30s) --------------------------------
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(30_000)

        QTimer.singleShot(0, self._refresh)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh(self):
        try:
            all_orders: list[PaperOrder] = self._paper.get_orders()
        except Exception as exc:
            self.header.set_subtitle(f"error: {exc}")
            return

        self._all_orders = all_orders

        open_orders = [o for o in all_orders if (getattr(o, "status", "") or "").upper() == "OPEN"]
        filled_today = self._filter_today_filled(all_orders)

        self._model_open.set_rows(open_orders)
        self._model_all.set_rows(all_orders)
        self._model_today.set_rows(filled_today)

        total = len(all_orders)
        open_count = len(open_orders)
        self.header.set_subtitle(
            f"{total} orders · {open_count} open · {len(filled_today)} filled today"
        )

    def _filter_today_filled(self, orders: list[PaperOrder]) -> list[PaperOrder]:
        """Return orders that are filled. Today filter is best-effort."""
        from datetime import date
        today = date.today().isoformat()
        result = []
        for o in orders:
            if (getattr(o, "status", "") or "").upper() != "FILLED":
                continue
            updated = getattr(o, "updated_at", None) or getattr(o, "created_at", None)
            if updated is not None and str(updated)[:10] == today:
                result.append(o)
        # If no today-specific filled orders, return all filled orders
        if not result:
            result = [o for o in orders if (getattr(o, "status", "") or "").upper() == "FILLED"]
        return result

    # ------------------------------------------------------------------
    # Row click handlers
    # ------------------------------------------------------------------

    def _on_open_row_clicked(self, idx: QModelIndex):
        if not idx.isValid():
            return
        order = self._model_open.row_at(idx.row())
        if order is not None:
            self._detail.set_order(order)
            sym = getattr(order, "symbol", None)
            if sym:
                self._link.set_symbol(sym)

    def _on_all_row_clicked(self, idx: QModelIndex):
        if not idx.isValid():
            return
        order = self._model_all.row_at(idx.row())
        if order is not None:
            self._detail.set_order(order)
            sym = getattr(order, "symbol", None)
            if sym:
                self._link.set_symbol(sym)

    def _on_today_row_clicked(self, idx: QModelIndex):
        if not idx.isValid():
            return
        order = self._model_today.row_at(idx.row())
        if order is not None:
            self._detail.set_order(order)
            sym = getattr(order, "symbol", None)
            if sym:
                self._link.set_symbol(sym)

    # ------------------------------------------------------------------
    # Trade dialog launchers
    # ------------------------------------------------------------------

    def _open_buy_dialog(self):
        symbol = self._link.context.symbol or ""
        dlg = TradeDialog(self._paper, mode="buy", symbol=symbol, parent=self)
        dlg.order_submitted.connect(self._on_order_submitted)
        dlg.exec()

    def _open_sell_dialog(self):
        symbol = self._link.context.symbol or ""
        dlg = TradeDialog(self._paper, mode="sell", symbol=symbol, parent=self)
        dlg.order_submitted.connect(self._on_order_submitted)
        dlg.exec()

    def _on_order_submitted(self, order: PaperOrder):
        self._refresh()
        # Highlight the new order in the All Orders tab
        self._tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Match orders
    # ------------------------------------------------------------------

    def _run_match(self):
        self._btn_match.setEnabled(False)
        self._btn_match.setText("Matching...")
        self.header.set_subtitle("Matching orders...")

        worker = _MatchWorker(self._paper)
        worker.signals.done.connect(self._on_match_done)
        worker.signals.error.connect(self._on_match_error)
        QThreadPool.globalInstance().start(worker)

    def _on_match_done(self, count: int):
        self._btn_match.setEnabled(True)
        self._btn_match.setText("Match ▶")
        self.header.set_subtitle(f"Matched {count} order{'s' if count != 1 else ''}")
        self._refresh()

    def _on_match_error(self, msg: str):
        self._btn_match.setEnabled(True)
        self._btn_match.setText("Match ▶")
        self.header.set_subtitle(f"Match error: {msg}")

    # ------------------------------------------------------------------
    # Cancel order
    # ------------------------------------------------------------------

    def _on_cancel_order(self, order: PaperOrder):
        try:
            order_id = getattr(order, "id", None) or getattr(order, "order_id", None)
            if order_id is not None:
                self._paper.cancel_order(order_id)
            self._detail.set_order(None)
            self._refresh()
        except Exception as exc:
            self.header.set_subtitle(f"Cancel error: {exc}")
