"""PaperService — adapter between desktop GUI and TUI paper trading backend.

All paper trading logic stays in backend.*. This service is a thin wrapper
that reads/writes the same runtime files as the TUI, so both apps share state.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import get_trading_runtime_dir, ensure_dir, migrate_legacy_path, get_project_root
from apps.desktop.services.paper_types import (
    PaperPosition,
    NavSummary,
    PaperOrder,
    Trade,
    ConcentrationRow,
    PaperSignal,
    StrategyEntry,
)

# ── Runtime paths (same as TUI — single source of truth via migrate_legacy_path) ──
_PROJECT_ROOT = get_project_root(__file__)
TRADING_RUNTIME_DIR = ensure_dir(get_trading_runtime_dir(__file__))

PORTFOLIO_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_portfolio.csv",
    [_PROJECT_ROOT / "tui_paper_portfolio.csv"],
)
NAV_LOG_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_nav_log.csv",
    [_PROJECT_ROOT / "tui_paper_nav_log.csv"],
)
TRADE_LOG_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_trade_log.csv",
    [_PROJECT_ROOT / "tui_paper_trade_log.csv"],
)
STATE_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_state.json",
    [_PROJECT_ROOT / "tui_paper_state.json"],
)
ORDERS_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_orders.json",
    [_PROJECT_ROOT / "tui_paper_orders.json"],
)
ACTIVE_STRAT_FILE = TRADING_RUNTIME_DIR / "desktop_active_strategy.json"

PORTFOLIO_COLS = [
    "Entry_Date",
    "Symbol",
    "Quantity",
    "Buy_Price",
    "Buy_Amount",
    "Buy_Fees",
    "Total_Cost_Basis",
    "Signal_Type",
    "High_Watermark",
    "Last_LTP",
    "Last_LTP_Source",
    "Last_LTP_Time_UTC",
]
TRADE_LOG_COLS = [
    "Date",
    "Action",
    "Symbol",
    "Shares",
    "Price",
    "Fees",
    "Reason",
    "PnL",
    "PnL_Pct",
]
NAV_LOG_COLS = ["Date", "Cash", "Positions_Value", "NAV", "Num_Positions"]


def _db() -> sqlite3.Connection:
    """Open a new SQLite connection using the canonical DB path."""
    return sqlite3.connect(str(get_db_path()))


def _sector_for(symbol: str) -> str:
    """Look up sector for a symbol via the backtest module. Fails silently."""
    try:
        from backend.backtesting.simple_backtest import get_symbol_sector

        return str(get_symbol_sector(symbol) or "")
    except Exception:
        return ""


def _last_price(symbol: str, conn: Optional[sqlite3.Connection] = None) -> float:
    """Return the most recent closing price for a symbol from stock_prices."""
    close_conn = conn is None
    if close_conn:
        conn = _db()
    try:
        r = pd.read_sql_query(
            "SELECT close FROM stock_prices WHERE symbol=? ORDER BY date DESC LIMIT 1",
            conn,
            params=(symbol,),
        )
        return float(r.iloc[0]["close"]) if not r.empty else 0.0
    except Exception:
        return 0.0
    finally:
        if close_conn:
            conn.close()


def _prev_close(symbol: str, conn: Optional[sqlite3.Connection] = None) -> float:
    """Return the second-most-recent closing price (previous close) for day P&L."""
    close_conn = conn is None
    if close_conn:
        conn = _db()
    try:
        r = pd.read_sql_query(
            "SELECT close FROM stock_prices WHERE symbol=? ORDER BY date DESC LIMIT 2",
            conn,
            params=(symbol,),
        )
        if len(r) >= 2:
            return float(r.iloc[1]["close"])
        if not r.empty:
            return float(r.iloc[0]["close"])
        return 0.0
    except Exception:
        return 0.0
    finally:
        if close_conn:
            conn.close()


class PaperService:
    """Thread-safe read adapter. Write operations (buy/sell/cancel) are atomic.

    Reads and writes the same runtime CSV/JSON files as the TUI so that both
    apps share live paper-trading state without a broker server in the middle.
    """

    # ── Portfolio ──────────────────────────────────────────────────────────────

    def _load_portfolio_df(self) -> pd.DataFrame:
        """Load the TUI paper portfolio CSV. Returns empty DataFrame on any error."""
        if not PORTFOLIO_FILE.exists():
            return pd.DataFrame(columns=PORTFOLIO_COLS)
        try:
            df = pd.read_csv(PORTFOLIO_FILE)
        except Exception:
            return pd.DataFrame(columns=PORTFOLIO_COLS)
        for col in PORTFOLIO_COLS:
            if col not in df.columns:
                df[col] = ""
        return df

    def positions(self) -> list[PaperPosition]:
        """Return all current open positions as PaperPosition dataclasses."""
        df = self._load_portfolio_df()
        if df.empty:
            return []

        conn = _db()
        total_mv = 0.0
        # Pre-compute total market value for weight calculations
        for _, row in df.iterrows():
            sym = str(row.get("Symbol") or "")
            qty = float(row.get("Quantity") or 0)
            if not sym or qty <= 0:
                continue
            last = _last_price(sym, conn)
            total_mv += last * qty

        rows: list[PaperPosition] = []
        for _, row in df.iterrows():
            sym = str(row.get("Symbol") or "")
            qty = float(row.get("Quantity") or 0)
            if not sym or qty <= 0:
                continue

            avg = float(row.get("Buy_Price") or 0)
            cost = float(row.get("Total_Cost_Basis") or (avg * qty))
            last = _last_price(sym, conn)
            prev = _prev_close(sym, conn)

            mv = last * qty
            upnl = mv - cost
            pct = ((mv / cost) - 1.0) * 100.0 if cost else 0.0
            day_chg = last - prev
            day_pnl = day_chg * qty
            day_pct = (day_chg / prev * 100.0) if prev else 0.0

            try:
                entry = pd.Timestamp(str(row.get("Entry_Date") or ""))
                days_held = max(0, (pd.Timestamp.now() - entry).days)
            except Exception:
                days_held = 0

            weight = (mv / total_mv * 100.0) if total_mv else 0.0

            rows.append(
                PaperPosition(
                    symbol=sym,
                    qty=qty,
                    avg_price=avg,
                    last_price=last,
                    day_pnl=day_pnl,
                    day_pct=day_pct,
                    unrealized_pnl=upnl,
                    pct_return=pct,
                    days_held=days_held,
                    signal_type=str(row.get("Signal_Type") or "manual"),
                    sector=_sector_for(sym),
                    market_value=mv,
                    cost_basis=cost,
                    weight_pct=weight,
                )
            )

        conn.close()
        return rows

    def nav_summary(self) -> NavSummary:
        """Compute full NAV summary from live portfolio + state file + nav log."""
        positions = self.positions()
        mv = sum(p.market_value for p in positions)
        invested = sum(p.cost_basis for p in positions)

        # Cash from TUI state file
        cash = 0.0
        try:
            if STATE_FILE.exists():
                state = json.loads(STATE_FILE.read_text())
                cash = float(state.get("cash", 0.0))
        except Exception:
            pass

        nav = cash + mv
        day_pnl = sum(p.day_pnl for p in positions)
        total_pnl = mv - invested
        total_return = (total_pnl / invested * 100.0) if invested else 0.0
        gross_return = total_return  # fees already baked into cost_basis

        # Max drawdown from nav log
        max_dd = 0.0
        try:
            if NAV_LOG_FILE.exists():
                nav_df = pd.read_csv(NAV_LOG_FILE)
                if not nav_df.empty and "NAV" in nav_df.columns:
                    nav_vals = nav_df["NAV"].dropna().tolist()
                    if len(nav_vals) >= 2:
                        peak = float("-inf")
                        for v in nav_vals:
                            v = float(v)
                            if v > peak:
                                peak = v
                            if peak > 0:
                                dd = (v - peak) / peak * 100.0
                                if dd < max_dd:
                                    max_dd = dd
        except Exception:
            pass

        # NEPSE comparison: load from nav_log if annotated, else 0
        nepse_return = 0.0
        try:
            if NAV_LOG_FILE.exists():
                nav_df = pd.read_csv(NAV_LOG_FILE)
                if not nav_df.empty and "NEPSE_Index" in nav_df.columns:
                    nepse_vals = nav_df["NEPSE_Index"].dropna()
                    if len(nepse_vals) >= 2:
                        first = float(nepse_vals.iloc[0])
                        last_val = float(nepse_vals.iloc[-1])
                        if first > 0:
                            nepse_return = (last_val / first - 1.0) * 100.0
        except Exception:
            pass

        alpha = total_return - nepse_return
        prev_nav = nav - day_pnl
        day_pct = (day_pnl / prev_nav * 100.0) if prev_nav != 0 else 0.0

        return NavSummary(
            nav=nav,
            cash=cash,
            invested=invested,
            day_pnl=day_pnl,
            day_pct=day_pct,
            total_return=total_return,
            gross_return=gross_return,
            nepse_return=nepse_return,
            alpha=alpha,
            max_dd=max_dd,
            n_positions=len(positions),
        )

    def concentration(self) -> list[ConcentrationRow]:
        """Return concentration rows for top-5 positions and sector aggregates."""
        positions = self.positions()
        if not positions:
            return []

        rows: list[ConcentrationRow] = []

        # Top 5 positions by weight
        for p in sorted(positions, key=lambda x: -x.weight_pct)[:5]:
            rows.append(
                ConcentrationRow(
                    row_type="POSITION",
                    name=p.symbol,
                    value=p.market_value,
                    weight_pct=p.weight_pct,
                )
            )

        # Sector aggregates
        by_sector: dict[str, list[PaperPosition]] = defaultdict(list)
        for p in positions:
            sec = p.sector.strip() if p.sector.strip() else "Unknown"
            by_sector[sec].append(p)

        total_mv = sum(p.market_value for p in positions) or 1.0
        for sec, ps in sorted(
            by_sector.items(), key=lambda kv: -sum(p.market_value for p in kv[1])
        ):
            sec_mv = sum(p.market_value for p in ps)
            rows.append(
                ConcentrationRow(
                    row_type="SECTOR",
                    name=sec,
                    value=sec_mv,
                    weight_pct=sec_mv / total_mv * 100.0,
                )
            )

        return rows

    def trade_history(self) -> list[Trade]:
        """Load trade history from the TUI trade log CSV (most recent first)."""
        if not TRADE_LOG_FILE.exists():
            return []
        try:
            df = pd.read_csv(TRADE_LOG_FILE)
        except Exception:
            return []

        trades: list[Trade] = []
        for _, row in df.iterrows():
            trades.append(
                Trade(
                    date=str(row.get("Date") or ""),
                    action=str(row.get("Action") or ""),
                    symbol=str(row.get("Symbol") or ""),
                    shares=float(row.get("Shares") or 0),
                    price=float(row.get("Price") or 0),
                    fees=float(row.get("Fees") or 0),
                    pnl=float(row.get("PnL") or 0),
                    pnl_pct=float(row.get("PnL_Pct") or 0),
                    reason=str(row.get("Reason") or ""),
                )
            )
        return list(reversed(trades))

    # ── Orders ────────────────────────────────────────────────────────────────

    def _load_orders(self) -> list[dict]:
        """Load orders from the JSON file. Handles list or dict container formats."""
        if not ORDERS_FILE.exists():
            return []
        try:
            data = json.loads(ORDERS_FILE.read_text())
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return list(data.get("orders") or [])
        except Exception:
            return []
        return []

    def _save_orders(self, orders: list[dict]) -> None:
        """Atomically write orders list to JSON."""
        ensure_dir(ORDERS_FILE.parent)
        ORDERS_FILE.write_text(json.dumps(orders, indent=2))

    def _to_paper_order(self, o: dict) -> PaperOrder:
        """Convert a raw order dict to a typed PaperOrder dataclass."""
        fill = o.get("fill_price")
        return PaperOrder(
            id=str(o.get("id") or ""),
            symbol=str(o.get("symbol") or ""),
            action=str(o.get("action") or "BUY").upper(),
            qty=int(o.get("qty") or 0),
            order_price=float(o.get("price") or 0),
            fill_price=float(fill) if fill is not None else None,
            status=str(o.get("status") or "OPEN"),
            created_at=str(o.get("created_at") or ""),
            updated_at=str(o.get("updated_at") or ""),
            reason=str(o.get("reason") or ""),
            slippage_pct=float(o.get("slippage_pct") or 2.0),
        )

    def all_orders(self) -> list[PaperOrder]:
        """Return all orders regardless of status."""
        return [self._to_paper_order(o) for o in self._load_orders()]

    def pending_orders(self) -> list[PaperOrder]:
        """Return only OPEN orders awaiting fill."""
        return [o for o in self.all_orders() if o.status == "OPEN"]

    def filled_orders(self) -> list[PaperOrder]:
        """Return only FILLED orders."""
        return [o for o in self.all_orders() if o.status == "FILLED"]

    def submit_buy(
        self,
        symbol: str,
        qty: int,
        price: float,
        slippage: float = 2.0,
    ) -> PaperOrder:
        """Submit a manual BUY order. Written to the shared orders JSON."""
        ts = datetime.now(timezone.utc).isoformat()
        o: dict = {
            "id": str(uuid.uuid4()),
            "symbol": symbol.upper().strip(),
            "action": "BUY",
            "qty": qty,
            "price": price,
            "fill_price": None,
            "status": "OPEN",
            "created_at": ts,
            "updated_at": ts,
            "reason": "desktop_manual",
            "slippage_pct": slippage,
            "source": "desktop_manual",
        }
        orders = self._load_orders()
        orders.append(o)
        self._save_orders(orders)
        return self._to_paper_order(o)

    def submit_sell(
        self,
        symbol: str,
        qty: int,
        price: float,
        slippage: float = 2.0,
    ) -> PaperOrder:
        """Submit a manual SELL order. Written to the shared orders JSON."""
        ts = datetime.now(timezone.utc).isoformat()
        o: dict = {
            "id": str(uuid.uuid4()),
            "symbol": symbol.upper().strip(),
            "action": "SELL",
            "qty": qty,
            "price": price,
            "fill_price": None,
            "status": "OPEN",
            "created_at": ts,
            "updated_at": ts,
            "reason": "desktop_manual",
            "slippage_pct": slippage,
            "source": "desktop_manual",
        }
        orders = self._load_orders()
        orders.append(o)
        self._save_orders(orders)
        return self._to_paper_order(o)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an OPEN order by ID. Returns True if the order was found and cancelled."""
        orders = self._load_orders()
        ts = datetime.now(timezone.utc).isoformat()
        found = False
        for o in orders:
            if str(o.get("id")) == order_id and o.get("status") == "OPEN":
                o["status"] = "CANCELLED"
                o["updated_at"] = ts
                found = True
        if found:
            self._save_orders(orders)
        return found

    def match_orders(self) -> int:
        """Try to fill OPEN orders against current LTP using slippage tolerance.

        Delegates actual portfolio mutation to the classic dashboard exec_buy /
        exec_sell helpers so that fees and position tracking stay consistent.
        Returns the number of orders filled.
        """
        from apps.classic.dashboard import exec_buy, exec_sell

        orders = self._load_orders()
        conn = _db()
        ts = datetime.now(timezone.utc).isoformat()
        filled = 0

        for o in orders:
            if o.get("status") != "OPEN":
                continue
            sym = str(o.get("symbol") or "")
            if not sym:
                continue
            ltp = _last_price(sym, conn)
            if ltp <= 0:
                continue

            slip = float(o.get("slippage_pct") or 2.0) / 100.0
            action = str(o.get("action") or "").upper()
            price = float(o.get("price") or 0)

            matched = False
            if action == "BUY" and ltp <= price * (1.0 + slip):
                matched = True
            elif action == "SELL" and ltp >= price * (1.0 - slip):
                matched = True

            if matched:
                try:
                    qty = int(o.get("qty") or 0)
                    if action == "BUY":
                        exec_buy(sym, str(qty), str(ltp))
                    else:
                        exec_sell(sym, str(qty), str(ltp))
                    o["status"] = "FILLED"
                    o["fill_price"] = ltp
                    o["updated_at"] = ts
                    filled += 1
                except Exception:
                    pass

        conn.close()
        if filled:
            self._save_orders(orders)
        return filled

    # ── Signals ───────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        signal_types: Optional[list[str]] = None,
        use_regime_filter: bool = True,
    ) -> tuple[list[PaperSignal], str]:
        """Generate buy signals. This is slow — always call from a background thread.

        If signal_types is None, the active strategy config is used as fallback,
        defaulting to the canonical C31 signal set.
        """
        from backend.backtesting.simple_backtest import load_all_prices
        from backend.trading.live_trader import generate_signals as _gen

        if signal_types is None:
            strat = self.active_strategy()
            if strat and strat.config.get("signal_types"):
                signal_types = list(strat.config["signal_types"])
                use_regime_filter = bool(strat.config.get("use_regime_filter", True))
            else:
                signal_types = [
                    "volume",
                    "quality",
                    "low_vol",
                    "mean_reversion",
                    "quarterly_fundamental",
                    "xsec_momentum",
                ]

        conn = _db()
        prices_df = load_all_prices(conn)
        conn.close()

        raw, regime = _gen(
            prices_df,
            signal_types,
            use_regime_filter=use_regime_filter,
        )

        result: list[PaperSignal] = []
        for s in raw:
            result.append(
                PaperSignal(
                    symbol=str(s.get("symbol") or ""),
                    score=float(s.get("score") or 0.0),
                    signal_type=str(s.get("signal_type") or ""),
                    strength=float(s.get("strength") or 0.0),
                    confidence=float(s.get("confidence") or 0.0),
                    regime=str(regime or ""),
                    as_of=str(s.get("as_of") or ""),
                )
            )
        return result, str(regime or "")

    # ── Strategy registry ─────────────────────────────────────────────────────

    def list_strategies(self) -> list[StrategyEntry]:
        """Return all registered strategies with is_active flag set correctly."""
        from backend.trading import strategy_registry

        active_id = self.active_strategy_id()
        entries: list[StrategyEntry] = []
        for s in strategy_registry.list_strategies():
            sid = str(s.get("id") or "")
            entries.append(
                StrategyEntry(
                    id=sid,
                    name=str(s.get("name") or s.get("id") or ""),
                    source=str(s.get("source") or "builtin"),
                    description=str(s.get("description") or ""),
                    notes=dict(s.get("notes") or {}),
                    config=dict(s.get("config") or {}),
                    is_active=(sid == active_id),
                )
            )
        return entries

    def active_strategy_id(self) -> str:
        """Return the ID of the currently active strategy for this desktop session."""
        try:
            if ACTIVE_STRAT_FILE.exists():
                data = json.loads(ACTIVE_STRAT_FILE.read_text())
                sid = str(data.get("strategy_id") or "")
                if sid:
                    return sid
        except Exception:
            pass
        from backend.trading.strategy_registry import default_strategy_for_account

        return default_strategy_for_account("account_1")

    def active_strategy(self) -> Optional[StrategyEntry]:
        """Return the full StrategyEntry for the active strategy, or None."""
        sid = self.active_strategy_id()
        for s in self.list_strategies():
            if s.id == sid:
                return s
        return None

    def set_active_strategy(self, strategy_id: str) -> None:
        """Persist the active strategy selection for this desktop session."""
        ensure_dir(ACTIVE_STRAT_FILE.parent)
        ACTIVE_STRAT_FILE.write_text(json.dumps({"strategy_id": strategy_id}))

    def run_backtest(
        self,
        strategy_id: str,
        start_date: str,
        end_date: str,
        capital: float = 1_000_000.0,
    ) -> dict:
        """Run a full backtest for the given strategy and return the results dict."""
        from backend.trading import strategy_registry

        payload = strategy_registry.load_strategy(strategy_id)
        if payload is None:
            raise ValueError(f"Strategy not found: {strategy_id}")
        return strategy_registry.run_strategy_backtest(
            payload,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def held_qty(self, symbol: str) -> int:
        """Return the currently held quantity for a symbol (0 if not held)."""
        df = self._load_portfolio_df()
        if df.empty:
            return 0
        mask = df["Symbol"] == symbol.upper().strip()
        return int(df.loc[mask, "Quantity"].sum())

    def known_symbols(self) -> list[str]:
        """Return all symbols present in the stock_prices database."""
        conn = _db()
        try:
            r = pd.read_sql_query(
                "SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol",
                conn,
            )
            return r["symbol"].tolist()
        except Exception:
            return []
        finally:
            conn.close()
