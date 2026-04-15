"""TradeDialog — modal buy/sell order entry for paper trading."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, QTimer, QThreadPool, QRunnable, QObject
from PySide6.QtGui import QColor, QIntValidator
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QFrame, QSizePolicy,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, BORDER_STRONG,
    TEXT, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT, GAIN, GAIN_HI, LOSS, LOSS_HI, WARN,
    mono_font, ui_font,
)
from apps.desktop.utils import fmt_number
from apps.desktop.services.paper_service import PaperService


# ---------------------------------------------------------------------------
# Background worker to fetch NAV summary (cash balance) without blocking UI
# ---------------------------------------------------------------------------

class _NavWorkerSignals(QObject):
    done = Signal(float)    # cash available
    error = Signal(str)


class _NavWorker(QRunnable):
    def __init__(self, paper: PaperService):
        super().__init__()
        self.paper = paper
        self.signals = _NavWorkerSignals()

    def run(self):
        try:
            nav = self.paper.nav_summary()
            self.signals.done.emit(float(nav.cash))
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Separator helper
# ---------------------------------------------------------------------------

def _sep() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER};")
    return f


# ---------------------------------------------------------------------------
# TradeDialog
# ---------------------------------------------------------------------------

class TradeDialog(QDialog):
    """Modal dialog for submitting paper BUY or SELL orders."""

    order_submitted = Signal(object)  # emits PaperOrder on success

    # Stylesheet constants for the action button
    _SS_BUY_BTN = (
        f"QPushButton {{"
        f"  background: transparent;"
        f"  color: {GAIN};"
        f"  border: 1px solid {GAIN};"
        f"  padding: 4px 18px;"
        f"  font-size: 12px;"
        f"  font-weight: 600;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background: {GAIN};"
        f"  color: {BG};"
        f"}}"
        f"QPushButton:pressed {{"
        f"  background: {GAIN_HI};"
        f"  color: {BG};"
        f"}}"
    )
    _SS_SELL_BTN = (
        f"QPushButton {{"
        f"  background: transparent;"
        f"  color: {LOSS};"
        f"  border: 1px solid {LOSS};"
        f"  padding: 4px 18px;"
        f"  font-size: 12px;"
        f"  font-weight: 600;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background: {LOSS};"
        f"  color: {BG};"
        f"}}"
        f"QPushButton:pressed {{"
        f"  background: {LOSS_HI};"
        f"  color: {BG};"
        f"}}"
    )
    _SS_CANCEL_BTN = (
        f"QPushButton {{"
        f"  background: transparent;"
        f"  color: {TEXT_SECONDARY};"
        f"  border: 1px solid {BORDER_STRONG};"
        f"  padding: 4px 14px;"
        f"  font-size: 12px;"
        f"}}"
        f"QPushButton:hover {{"
        f"  border-color: {TEXT_SECONDARY};"
        f"  color: {TEXT};"
        f"}}"
    )
    _SS_INPUT = (
        f"QLineEdit {{"
        f"  background: {BG};"
        f"  color: {TEXT};"
        f"  border: 1px solid {BORDER};"
        f"  padding: 4px 8px;"
        f"  font-family: 'JetBrains Mono', 'IBM Plex Mono', 'Menlo', 'Monaco', 'Courier New', monospace;"
        f"  font-size: 13px;"
        f"}}"
        f"QLineEdit:focus {{"
        f"  border: 1px solid {ACCENT};"
        f"}}"
        f"QLineEdit:disabled {{"
        f"  color: {TEXT_MUTED};"
        f"}}"
    )

    def __init__(
        self,
        paper: PaperService,
        mode: str,
        symbol: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._paper = paper
        self._mode = mode.lower().strip()  # "buy" or "sell"
        self._is_buy = self._mode == "buy"
        self._held_qty: int = 0
        self._cash_available: float = 0.0
        self._last_price: Optional[float] = None

        self.setWindowTitle("Paper Trade — " + ("BUY" if self._is_buy else "SELL"))
        self.setFixedWidth(380)
        self.setModal(True)
        self.setStyleSheet(
            f"QDialog {{ background: {ELEVATED}; }}"
            f"QLabel {{ background: transparent; }}"
        )

        self._build_ui(symbol.strip().upper())
        self._connect_signals()

        # Fetch available balance / held qty in background after dialog opens
        QTimer.singleShot(0, self._fetch_context)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, symbol: str):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        # ---- Title ---------------------------------------------------
        mode_color = GAIN if self._is_buy else LOSS
        mode_label = "● BUY ORDER" if self._is_buy else "● SELL ORDER"
        self._title = QLabel(mode_label)
        self._title.setFont(ui_font(15, 700))
        self._title.setStyleSheet(f"color: {mode_color}; letter-spacing: 0.5px;")
        root.addWidget(self._title)

        # ---- Form grid -----------------------------------------------
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)
        form.setColumnStretch(1, 1)

        def _lbl(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setFont(ui_font(11, 500))
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 0.3px;")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lbl

        # Symbol
        form.addWidget(_lbl("Symbol"), 0, 0)
        self._sym_edit = QLineEdit(symbol)
        self._sym_edit.setPlaceholderText("e.g. NABIL")
        self._sym_edit.setFont(mono_font(13))
        self._sym_edit.setStyleSheet(self._SS_INPUT)
        self._sym_edit.setFixedHeight(30)
        form.addWidget(self._sym_edit, 0, 1)

        # Shares
        form.addWidget(_lbl("Shares"), 1, 0)
        shares_col = QHBoxLayout()
        shares_col.setSpacing(6)
        self._qty_edit = QLineEdit()
        self._qty_edit.setPlaceholderText("100" if self._is_buy else "all")
        self._qty_edit.setFont(mono_font(13))
        self._qty_edit.setStyleSheet(self._SS_INPUT)
        self._qty_edit.setFixedHeight(30)
        shares_col.addWidget(self._qty_edit, 1)
        self._qty_hint = QLabel("")
        self._qty_hint.setFont(ui_font(10))
        self._qty_hint.setStyleSheet(f"color: {TEXT_MUTED};")
        shares_col.addWidget(self._qty_hint, 0)
        shares_wrap = QFrame()
        shares_wrap.setStyleSheet("QFrame { background: transparent; }")
        shares_wrap.setLayout(shares_col)
        form.addWidget(shares_wrap, 1, 1)

        # Price
        form.addWidget(_lbl("Price"), 2, 0)
        price_col = QHBoxLayout()
        price_col.setSpacing(6)
        self._price_edit = QLineEdit()
        self._price_edit.setPlaceholderText("last close")
        self._price_edit.setFont(mono_font(13))
        self._price_edit.setStyleSheet(self._SS_INPUT)
        self._price_edit.setFixedHeight(30)
        price_col.addWidget(self._price_edit, 1)
        opt_lbl = QLabel("optional")
        opt_lbl.setFont(ui_font(10))
        opt_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        price_col.addWidget(opt_lbl, 0)
        price_wrap = QFrame()
        price_wrap.setStyleSheet("QFrame { background: transparent; }")
        price_wrap.setLayout(price_col)
        form.addWidget(price_wrap, 2, 1)

        # Slippage
        form.addWidget(_lbl("Slippage"), 3, 0)
        slip_col = QHBoxLayout()
        slip_col.setSpacing(6)
        self._slip_edit = QLineEdit("2.0")
        self._slip_edit.setFont(mono_font(13))
        self._slip_edit.setStyleSheet(self._SS_INPUT)
        self._slip_edit.setFixedHeight(30)
        self._slip_edit.setMaximumWidth(80)
        slip_col.addWidget(self._slip_edit)
        pct_lbl = QLabel("%")
        pct_lbl.setFont(mono_font(12))
        pct_lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        slip_col.addWidget(pct_lbl)
        slip_col.addStretch(1)
        slip_wrap = QFrame()
        slip_wrap.setStyleSheet("QFrame { background: transparent; }")
        slip_wrap.setLayout(slip_col)
        form.addWidget(slip_wrap, 3, 1)

        root.addLayout(form)
        root.addWidget(_sep())

        # ---- Estimate row --------------------------------------------
        est_grid = QGridLayout()
        est_grid.setHorizontalSpacing(8)
        est_grid.setVerticalSpacing(5)

        def _kl(text: str) -> QLabel:
            l = QLabel(text)
            l.setFont(ui_font(11))
            l.setStyleSheet(f"color: {TEXT_MUTED};")
            return l

        def _vl(text: str = "—", color: str = TEXT) -> QLabel:
            l = QLabel(text)
            l.setFont(mono_font(12, 500))
            l.setStyleSheet(f"color: {color};")
            return l

        est_grid.addWidget(_kl("Est. Cost:"), 0, 0)
        self._est_label = _vl("—", TEXT_SECONDARY)
        est_grid.addWidget(self._est_label, 0, 1)

        avail_key_text = "Available:" if self._is_buy else "Holding:"
        est_grid.addWidget(_kl(avail_key_text), 1, 0)
        self._avail_label = _vl("loading...", TEXT_MUTED)
        est_grid.addWidget(self._avail_label, 1, 1)

        root.addLayout(est_grid)

        # ---- Error / status label ------------------------------------
        self._err_label = QLabel("")
        self._err_label.setFont(ui_font(11))
        self._err_label.setStyleSheet(f"color: {TEXT_MUTED};")
        self._err_label.setWordWrap(True)
        self._err_label.setFixedHeight(18)
        self._err_label.hide()
        root.addWidget(self._err_label)

        # ---- Buttons -------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFont(ui_font(12))
        self._cancel_btn.setFixedHeight(30)
        self._cancel_btn.setStyleSheet(self._SS_CANCEL_BTN)
        btn_row.addWidget(self._cancel_btn)

        submit_label = "Submit BUY ▶" if self._is_buy else "Submit SELL ▶"
        self._submit_btn = QPushButton(submit_label)
        self._submit_btn.setFont(ui_font(12, 600))
        self._submit_btn.setFixedHeight(30)
        self._submit_btn.setStyleSheet(
            self._SS_BUY_BTN if self._is_buy else self._SS_SELL_BTN
        )
        btn_row.addWidget(self._submit_btn)

        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        # Force uppercase on symbol field
        self._sym_edit.textChanged.connect(self._on_symbol_changed)

        # Live estimate updates
        self._qty_edit.textChanged.connect(self._update_estimate)
        self._price_edit.textChanged.connect(self._update_estimate)
        self._slip_edit.textChanged.connect(self._update_estimate)

        # Buttons
        self._cancel_btn.clicked.connect(self.reject)
        self._submit_btn.clicked.connect(self._on_submit)

    # ------------------------------------------------------------------
    # Slot: symbol uppercase enforcement
    # ------------------------------------------------------------------

    def _on_symbol_changed(self, text: str):
        upper = text.upper()
        if upper != text:
            cur = self._sym_edit.cursorPosition()
            self._sym_edit.blockSignals(True)
            self._sym_edit.setText(upper)
            self._sym_edit.setCursorPosition(cur)
            self._sym_edit.blockSignals(False)

    # ------------------------------------------------------------------
    # Background context fetch
    # ------------------------------------------------------------------

    def _fetch_context(self):
        symbol = self._sym_edit.text().strip().upper()
        if not self._is_buy:
            # Fetch held quantity for the symbol
            try:
                positions = self._paper.positions()
                held = 0
                for pos in positions:
                    if pos.symbol == symbol:
                        held = int(pos.quantity)
                        break
                self._held_qty = held
                self._avail_label.setText(
                    f"{held:,} shares" if held > 0 else "0 shares (none held)"
                )
                if held > 0:
                    self._qty_hint.setText(f"(all = {held:,} shares)")
                else:
                    self._qty_hint.setText("(no position)")
            except Exception as exc:
                self._avail_label.setText(f"error: {exc}")
        else:
            # Fetch cash available
            worker = _NavWorker(self._paper)
            worker.signals.done.connect(self._on_nav_loaded)
            worker.signals.error.connect(self._on_nav_error)
            QThreadPool.globalInstance().start(worker)

    def _on_nav_loaded(self, cash: float):
        self._cash_available = cash
        self._avail_label.setText(f"NPR {fmt_number(cash)}")
        self._avail_label.setStyleSheet(f"color: {TEXT};")
        self._update_estimate()

    def _on_nav_error(self, msg: str):
        self._avail_label.setText(f"error: {msg}")
        self._avail_label.setStyleSheet(f"color: {WARN};")

    # ------------------------------------------------------------------
    # Live estimate calculation
    # ------------------------------------------------------------------

    def _update_estimate(self):
        # Resolve price
        price_text = self._price_edit.text().strip()
        if price_text:
            try:
                price = float(price_text)
                if price <= 0:
                    raise ValueError("non-positive price")
            except ValueError:
                self._est_label.setText("invalid price")
                self._est_label.setStyleSheet(f"color: {WARN};")
                return
        else:
            # Use last close from service if available
            sym = self._sym_edit.text().strip().upper()
            if sym and self._last_price is None:
                try:
                    snap = self._paper.get_last_price(sym)
                    self._last_price = float(snap) if snap is not None else None
                except Exception:
                    self._last_price = None
            price = self._last_price

        if price is None:
            self._est_label.setText("— (price unknown)")
            self._est_label.setStyleSheet(f"color: {TEXT_MUTED};")
            return

        # Resolve qty
        qty_text = self._qty_edit.text().strip()
        if not qty_text:
            self._est_label.setText("—")
            self._est_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
            return

        if qty_text.lower() == "all":
            qty = self._held_qty
        else:
            try:
                qty = int(qty_text)
                if qty <= 0:
                    raise ValueError("non-positive qty")
            except ValueError:
                self._est_label.setText("invalid qty")
                self._est_label.setStyleSheet(f"color: {WARN};")
                return

        # Resolve slippage
        slip_text = self._slip_edit.text().strip()
        try:
            slip = float(slip_text) if slip_text else 2.0
        except ValueError:
            slip = 2.0

        effective_price = price * (1 + slip / 100.0) if self._is_buy else price * (1 - slip / 100.0)
        total = effective_price * qty
        self._est_label.setText(
            f"NPR {fmt_number(effective_price)} × {qty:,} = NPR {fmt_number(total)}"
        )
        self._est_label.setStyleSheet(f"color: {TEXT};")

    # ------------------------------------------------------------------
    # Error display helpers
    # ------------------------------------------------------------------

    def _show_error(self, msg: str):
        self._err_label.setText(msg)
        self._err_label.setStyleSheet(f"color: {LOSS};")
        self._err_label.show()

    def _clear_error(self):
        self._err_label.setText("")
        self._err_label.hide()

    # ------------------------------------------------------------------
    # Submit handler
    # ------------------------------------------------------------------

    def _on_submit(self):
        self._clear_error()

        # --- Validate symbol ---
        symbol = self._sym_edit.text().strip().upper()
        if not symbol:
            self._show_error("Symbol is required.")
            self._sym_edit.setFocus()
            return

        # --- Validate shares ---
        qty_text = self._qty_edit.text().strip()
        if not qty_text:
            self._show_error("Shares quantity is required.")
            self._qty_edit.setFocus()
            return

        if qty_text.lower() == "all":
            if not self._is_buy:
                qty = self._held_qty
                if qty <= 0:
                    self._show_error(f"No shares of {symbol} held — cannot sell.")
                    return
            else:
                self._show_error("'all' is only valid for SELL orders.")
                return
        else:
            try:
                qty = int(qty_text)
                if qty <= 0:
                    raise ValueError()
            except ValueError:
                self._show_error("Shares must be a positive integer (or 'all' for sell).")
                self._qty_edit.setFocus()
                return

        # --- Sell: check held quantity ---
        if not self._is_buy:
            if qty > self._held_qty:
                self._show_error(
                    f"Cannot sell {qty:,} — only {self._held_qty:,} shares held."
                )
                return

        # --- Validate price ---
        price_text = self._price_edit.text().strip()
        price: Optional[float] = None
        if price_text:
            try:
                price = float(price_text)
                if price <= 0:
                    raise ValueError()
            except ValueError:
                self._show_error("Price must be a positive number (or leave blank for last close).")
                self._price_edit.setFocus()
                return

        # --- Validate slippage ---
        slip_text = self._slip_edit.text().strip()
        try:
            slip = float(slip_text) if slip_text else 2.0
            if not (0.0 <= slip <= 10.0):
                raise ValueError()
        except ValueError:
            self._show_error("Slippage must be between 0 and 10 (%).")
            self._slip_edit.setFocus()
            return

        # --- Disable submit button to prevent double-click ---
        self._submit_btn.setEnabled(False)
        self._submit_btn.setText("Submitting...")

        try:
            if self._is_buy:
                order = self._paper.submit_buy(
                    symbol=symbol,
                    qty=qty,
                    price=price,
                    slippage=slip,
                )
            else:
                order = self._paper.submit_sell(
                    symbol=symbol,
                    qty=qty,
                    price=price,
                    slippage=slip,
                )
            self.order_submitted.emit(order)
            self.accept()
        except Exception as exc:
            self._show_error(f"Submit failed: {exc}")
            self._submit_btn.setEnabled(True)
            self._submit_btn.setText(
                "Submit BUY ▶" if self._is_buy else "Submit SELL ▶"
            )
