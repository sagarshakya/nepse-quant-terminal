"""Top-level main window: nav rail + workspace stack + ticker strip."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget,
)

from apps.desktop.theme import BG, BORDER, PANE, TEXT_SECONDARY, ui_font
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.nav_rail import NavRail
from apps.desktop.widgets.ticker_strip import TickerStrip
from apps.desktop.widgets.command_palette import CommandPalette, Command
from apps.desktop.widgets.trade_dialog import TradeDialog
from apps.desktop.workspaces.ticker_deep_dive import TickerDeepDive
from apps.desktop.workspaces.dashboard import Dashboard
from apps.desktop.workspaces.portfolio import Portfolio
from apps.desktop.workspaces.signals import Signals
from apps.desktop.workspaces.backtests import Backtests
from apps.desktop.workspaces.strategies import Strategies
from apps.desktop.workspaces.market_overview import MarketOverview
from apps.desktop.services.paper_service import PaperService
from backend.core.services import MarketService, BacktestService, SignalService, PortfolioService


# ---------------------------------------------------------------------------
# Workspace registry
# (key, display-title, nav-glyph, tooltip)
# Order is significant — Ctrl+1..8 shortcuts match this order.
# ---------------------------------------------------------------------------

WORKSPACES = [
    ("dashboard",  "Dashboard",  "⊞",  "Dashboard  (Ctrl+1)"),
    ("ticker",     "Price",      "◉",  "Price  (Ctrl+2)"),
    ("portfolio",  "Portfolio",  "▤",  "Portfolio  (Ctrl+3)"),
    ("orders",     "Orders",     "⊡",  "Orders  (Ctrl+4)"),
    ("signals",    "Signals",    "≋",  "Signals  (Ctrl+5)"),
    ("strategies", "Strategy",   "⚑",  "Strategy  (Ctrl+6)"),
    ("backtests",  "Backtests",  "⟁",  "Backtests  (Ctrl+7)"),
    ("market",     "Market",     "⊠",  "Market  (Ctrl+8)"),
]


# ---------------------------------------------------------------------------
# Minimal Orders workspace (standalone; a full workspace lives in orders.py
# once it exists — this fallback keeps the app functional without that file)
# ---------------------------------------------------------------------------

def _make_orders_workspace(paper: PaperService, link: LinkGroup) -> QWidget:
    """Return the Orders workspace.  Imports the real module if it exists,
    otherwise returns a placeholder frame so the app still launches."""
    try:
        from apps.desktop.workspaces.orders import Orders
        return Orders(paper, link)
    except ImportError:
        placeholder = QFrame()
        placeholder.setStyleSheet(f"QFrame {{ background: {BG}; }}")
        from PySide6.QtWidgets import QLabel, QVBoxLayout
        lay = QVBoxLayout(placeholder)
        lbl = QLabel("Orders workspace coming soon.\nCreate apps/desktop/workspaces/orders.py")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(ui_font(13))
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        return placeholder


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nepse Quant Workstation")
        self.resize(1440, 900)

        # ---- services --------------------------------------------------------
        self.market    = MarketService()
        self.backtests = BacktestService()
        self.signals   = SignalService()
        self.portfolio = PortfolioService(self.market)
        self.paper     = PaperService()

        # ---- link groups (one primary) ---------------------------------------
        self.link_a = LinkGroup(name="A", color="#4D9FFF")

        # ---- main composition ------------------------------------------------
        central = QWidget()
        central.setStyleSheet(f"QWidget {{ background: {BG}; }}")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Ticker strip across the top
        self.ticker_strip = TickerStrip(self.market)
        outer.addWidget(self.ticker_strip)

        # Body = nav rail + workspace stack
        body = QFrame()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # Build nav rail with all workspace items
        nav_items = [(key, glyph, tip) for key, _title, glyph, tip in WORKSPACES]
        self.nav_rail = NavRail(items=nav_items)
        self.nav_rail.workspace_selected.connect(self._select_workspace)
        body_lay.addWidget(self.nav_rail)

        # Workspace stack
        self.workspace_stack = QStackedWidget()

        # Construct all workspace widgets
        self.ws_dashboard  = Dashboard(self.market, self.signals, self.link_a)
        self.ws_ticker     = TickerDeepDive(self.market, self.link_a)
        self.ws_portfolio  = Portfolio(self.paper, self.link_a)
        self.ws_orders     = _make_orders_workspace(self.paper, self.link_a)
        self.ws_signals    = Signals(self.paper, self.market, self.link_a)
        self.ws_strategies = Strategies(self.paper, self.link_a)
        self.ws_backtests  = Backtests(self.backtests)
        self.ws_market     = MarketOverview(self.market, self.link_a)

        # Register in the stack; store index per key
        self._ws_indices: dict[str, int] = {
            "dashboard":  self.workspace_stack.addWidget(self.ws_dashboard),
            "ticker":     self.workspace_stack.addWidget(self.ws_ticker),
            "portfolio":  self.workspace_stack.addWidget(self.ws_portfolio),
            "orders":     self.workspace_stack.addWidget(self.ws_orders),
            "signals":    self.workspace_stack.addWidget(self.ws_signals),
            "strategies": self.workspace_stack.addWidget(self.ws_strategies),
            "backtests":  self.workspace_stack.addWidget(self.ws_backtests),
            "market":     self.workspace_stack.addWidget(self.ws_market),
        }

        body_lay.addWidget(self.workspace_stack, 1)
        outer.addWidget(body, 1)

        self.setCentralWidget(central)

        # ---- command palette -------------------------------------------------
        self.palette = CommandPalette(self)

        # ---- keyboard shortcuts ----------------------------------------------
        self._install_shortcuts()

        # Rebuild palette after startup (needs service data)
        QTimer.singleShot(50, self._rebuild_palette)

        # Load default symbol and navigate to dashboard
        QTimer.singleShot(100, self._load_default_symbol)

    # ------------------------------------------------------------------
    # Default symbol / workspace
    # ------------------------------------------------------------------

    def _load_default_symbol(self) -> None:
        self._select_workspace("dashboard")
        preferred = ["NEPSE", "NABIL", "NRIC", "NICA", "ADBL"]
        syms = set(self.market.symbols(include_indices=True))
        for s in preferred:
            if s in syms:
                self.link_a.set_symbol(s)
                return
        any_syms = self.market.symbols()
        if any_syms:
            self.link_a.set_symbol(any_syms[0])

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------

    def _install_shortcuts(self) -> None:
        def bind(seq: str, callback) -> None:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(callback)

        # Command palette
        bind("Ctrl+K",  self._open_palette)
        bind("Meta+K",  self._open_palette)

        # Ticker jump
        bind("Ctrl+T",  self._focus_ticker_jump)
        bind("Meta+T",  self._focus_ticker_jump)

        # Grid filter
        bind("Ctrl+F",  self._focus_grid_filter)
        bind("Meta+F",  self._focus_grid_filter)
        bind("/",       self._focus_grid_filter)

        # Workspace shortcuts Ctrl+1..8 (and Meta equivalents for macOS)
        for i, (key, *_rest) in enumerate(WORKSPACES, start=1):
            bind(f"Ctrl+{i}",  lambda checked=False, k=key: self._select_workspace(k))
            bind(f"Meta+{i}",  lambda checked=False, k=key: self._select_workspace(k))

        # Global buy / sell shortcuts
        bind("b", lambda: self._open_trade_dialog("buy"))
        bind("s", lambda: self._open_trade_dialog("sell"))

    # ------------------------------------------------------------------
    # Workspace management
    # ------------------------------------------------------------------

    def _select_workspace(self, key: str) -> None:
        if key not in self._ws_indices:
            return
        self.workspace_stack.setCurrentIndex(self._ws_indices[key])
        self.nav_rail.set_active(key)

    # ------------------------------------------------------------------
    # Grid filter
    # ------------------------------------------------------------------

    def _focus_grid_filter(self) -> None:
        ws = self.workspace_stack.currentWidget()
        if hasattr(ws, "grid"):
            ws.grid.focus_filter()
        elif hasattr(ws, "filter_edit"):
            ws.filter_edit.setFocus()

    # ------------------------------------------------------------------
    # Trade dialog
    # ------------------------------------------------------------------

    def _open_trade_dialog(self, mode: str) -> None:
        sym = self.link_a.context.symbol or ""
        dlg = TradeDialog(self.paper, mode=mode, symbol=sym, parent=self)
        dlg.order_submitted.connect(self._on_order_submitted)
        dlg.exec()

    def _on_order_submitted(self, order) -> None:
        # Refresh orders and portfolio workspaces if they support it
        if hasattr(self.ws_orders, "_refresh"):
            self.ws_orders._refresh()
        if hasattr(self.ws_portfolio, "_refresh"):
            self.ws_portfolio._refresh()

    # ------------------------------------------------------------------
    # Command palette
    # ------------------------------------------------------------------

    def _focus_ticker_jump(self) -> None:
        self._open_palette(prefill=":")

    def _open_palette(self, prefill: str = "") -> None:
        self._rebuild_palette()
        self.palette.open()
        if prefill:
            self.palette.input.setText(prefill)

    def _rebuild_palette(self) -> None:
        commands: list[Command] = []

        # Workspace navigation commands
        for i, (key, title, _glyph, _tip) in enumerate(WORKSPACES, start=1):
            commands.append(Command(
                label=f"Open {title}",
                hint=f"workspace · ctrl+{i}",
                action=lambda k=key: self._select_workspace(k),
            ))

        # Buy / sell shortcuts
        commands.append(Command(
            label="Buy — open trade dialog",
            hint="shortcut · b",
            action=lambda: self._open_trade_dialog("buy"),
        ))
        commands.append(Command(
            label="Sell — open trade dialog",
            hint="shortcut · s",
            action=lambda: self._open_trade_dialog("sell"),
        ))

        # Ticker jump commands
        syms = self.market.symbols()
        for sym in syms:
            commands.append(Command(
                label=sym,
                hint="ticker jump",
                action=lambda s=sym: self._jump_ticker(s),
                kind="ticker",
            ))

        self.palette.set_commands(commands)

    def _jump_ticker(self, symbol: str) -> None:
        self.link_a.set_symbol(symbol)
        self._select_workspace("ticker")
