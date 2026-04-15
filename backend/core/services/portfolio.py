"""PortfolioService — read-only projection over portfolio_positions + latest quotes."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from backend.quant_pro.database import get_db_path
from backend.core.types import Position
from backend.core.services.market import MarketService


class PortfolioService:
    def __init__(self, market: Optional[MarketService] = None, db_path: Optional[Path] = None):
        self._db_path = Path(db_path) if db_path is not None else Path(get_db_path())
        self._market = market or MarketService(self._db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def positions(self) -> list[Position]:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT symbol, quantity, avg_price FROM portfolio_positions"
                )
                raw = cur.fetchall()
        except sqlite3.Error:
            return []

        if not raw:
            return []

        symbols = [r["symbol"] for r in raw]
        snap = self._market.snapshot(symbols)
        by_sym = {q.symbol: q for q in snap.quotes}

        positions: list[Position] = []
        total_mv = 0.0
        for r in raw:
            sym = r["symbol"]
            qty = float(r["quantity"] or 0.0)
            avg = float(r["avg_price"] or 0.0)
            last = by_sym[sym].last if sym in by_sym else avg
            mv = qty * last
            pnl = (last - avg) * qty
            pnl_pct = ((last / avg) - 1.0) * 100.0 if avg else 0.0
            total_mv += mv
            positions.append(Position(
                symbol=sym, quantity=qty, avg_price=avg, last_price=last,
                market_value=mv, unrealized_pnl=pnl, unrealized_pnl_pct=pnl_pct,
                weight_pct=0.0,
            ))
        if total_mv > 0:
            for p in positions:
                p.weight_pct = p.market_value / total_mv * 100.0
        return positions
