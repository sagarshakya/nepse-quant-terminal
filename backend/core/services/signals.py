"""SignalService — read-only access to recent signal fires.

Phase 0 scope: surface rows from `broker_signals_v2` (if present) for the
Signals pane. Live factor recomputation is deferred; the service returns
whatever the existing pipeline already persisted.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from backend.quant_pro.database import get_db_path
from backend.core.types import Signal


class SignalService:
    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = Path(db_path) if db_path is not None else Path(get_db_path())

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def recent(self, *, limit: int = 200) -> list[Signal]:
        sql = """
            SELECT symbol, signal_date, signal_name, score, regime
            FROM broker_signals_v2
            ORDER BY signal_date DESC, score DESC
            LIMIT ?
        """
        rows: list[Signal] = []
        try:
            with self._connect() as conn:
                cur = conn.execute(sql, (limit,))
                for r in cur.fetchall():
                    try:
                        d = datetime.strptime(r["signal_date"], "%Y-%m-%d").date()
                    except Exception:
                        d = date.today()
                    rows.append(Signal(
                        symbol=r["symbol"],
                        name=r["signal_name"] or "",
                        score=float(r["score"] or 0.0),
                        as_of=d,
                        regime=(r["regime"] or "") if "regime" in r.keys() else "",
                    ))
        except sqlite3.Error:
            # Table may not exist in this install — return empty list.
            return []
        return rows
