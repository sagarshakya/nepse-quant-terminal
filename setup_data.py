#!/usr/bin/env python3
"""
NEPSE Quant Terminal — Initial data setup.

Fetches historical OHLCV price data for all NEPSE symbols from Merolagani
and populates the local nepse_data.db database.

Usage:
    python setup_data.py              # full backfill (all symbols, 2+ years)
    python setup_data.py --days 90   # quick test with 90 days
    python setup_data.py --symbol NABIL  # single symbol

Typically takes 20-60 minutes for a full backfill (throttled to respect
Merolagani rate limits). Run once after cloning, then use daily_update.py
for incremental updates.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from backend.quant_pro.database import get_db_path, init_db, save_to_db
from backend.quant_pro.vendor_api import fetch_ohlcv_chunk

ALL_SYMBOLS_FILE = Path(__file__).parent / "all_symbols.txt"
DELAY_BETWEEN_SYMBOLS = 1.2  # seconds — stay under Merolagani rate limit


def load_symbols() -> list[str]:
    if ALL_SYMBOLS_FILE.exists():
        syms = [s.strip() for s in ALL_SYMBOLS_FILE.read_text().splitlines() if s.strip()]
        # Exclude index/sector proxies
        return [s for s in syms if not s.startswith("SECTOR::") and s != "NEPSE"]
    return []


def ts(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def backfill_symbol(symbol: str, start: datetime, end: datetime) -> int:
    """Fetch OHLCV for one symbol and write to DB. Returns rows saved."""
    try:
        df = fetch_ohlcv_chunk(symbol, ts(start), ts(end))
        if df is None or df.empty:
            return 0
        df["symbol"] = symbol
        save_to_db(df, symbol)
        return len(df)
    except Exception as e:
        print(f"  WARN {symbol}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Backfill NEPSE price data")
    parser.add_argument("--days", type=int, default=760, help="Days of history to fetch (default 760 ≈ 2yr)")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to fetch")
    args = parser.parse_args()

    print("Initialising database...")
    init_db()

    end   = datetime.now()
    start = end - timedelta(days=args.days)

    symbols = [args.symbol] if args.symbol else load_symbols()
    if not symbols:
        print("ERROR: no symbols found. Check all_symbols.txt exists.")
        return

    print(f"Fetching {len(symbols)} symbols  |  {start.date()} → {end.date()}")
    print(f"DB: {get_db_path()}\n")

    total_rows = 0
    for i, sym in enumerate(symbols, 1):
        rows = backfill_symbol(sym, start, end)
        total_rows += rows
        print(f"[{i:3d}/{len(symbols)}] {sym:<12} {rows:>5} rows")
        if i < len(symbols):
            time.sleep(DELAY_BETWEEN_SYMBOLS)

    print(f"\nDone — {total_rows:,} rows written to {get_db_path()}")
    print("Run `python3 -m apps.tui.dashboard_tui` to launch the terminal.")


if __name__ == "__main__":
    main()
