"""
Gold and Silver price ingestion from yfinance (international markets).

Downloads XAU/USD and XAG/USD daily OHLCV data and stores in the
macro_indicators table alongside existing remittance and NRB rate data.
Also fetches USD/NPR exchange rate to enable local-price conversion.

Indicators stored (indicator_name → description):
    gold_usd_per_oz       — XAU/USD close price (USD per troy oz)
    silver_usd_per_oz     — XAG/USD close price (USD per troy oz)
    usd_npr_rate          — USD/NPR exchange rate (NPR per 1 USD)
    gold_return           — Daily log return of gold (decimal, e.g. 0.0152)
    silver_return         — Daily log return of silver (decimal)

Data sources:
    yfinance tickers:
        GC=F  — COMEX Gold Futures (front month, most liquid)
        SI=F  — COMEX Silver Futures (front month)
        USDNPR=X — USD/NPR forex pair (fallback: hardcoded NRB approximations)

Usage:
    python3 -m backend.quant_pro.data_scrapers.gold_silver_ingestion
    python3 -m backend.quant_pro.data_scrapers.gold_silver_ingestion --years 5
    python3 -m backend.quant_pro.data_scrapers.gold_silver_ingestion --db data/nepse_market_data.db
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.quant_pro.paths import get_data_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(get_data_dir(__file__) / "nepse_market_data.db")

# --------------------------------------------------------------------------
# Fallback USD/NPR rates (NRB interbank mid-rate, annual approximations)
# Used if yfinance cannot fetch USDNPR=X
# --------------------------------------------------------------------------
_FALLBACK_USDNPR: List[Tuple[str, float]] = [
    ("2018-01-01", 102.0),
    ("2019-01-01", 112.0),
    ("2020-01-01", 115.5),
    ("2021-01-01", 118.0),
    ("2022-01-01", 119.0),
    ("2022-07-01", 127.0),  # NPR weakened significantly mid-2022
    ("2023-01-01", 132.5),
    ("2023-07-01", 133.0),
    ("2024-01-01", 133.5),
    ("2024-07-01", 134.0),
    ("2025-01-01", 134.5),
    ("2026-01-01", 135.0),
]


def _get_fallback_rate(date_str: str) -> float:
    """Return closest-on-or-before fallback USD/NPR rate for a given date."""
    rate = _FALLBACK_USDNPR[0][1]
    for d, r in _FALLBACK_USDNPR:
        if d <= date_str:
            rate = r
        else:
            break
    return rate


def _ensure_macro_table(conn: sqlite3.Connection) -> None:
    """Create macro_indicators table if it doesn't exist (matches existing schema)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_indicators (
            date TEXT NOT NULL,
            indicator_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            period TEXT,
            source TEXT,
            publication_date TEXT,
            scraped_at_utc TEXT NOT NULL,
            PRIMARY KEY (date, indicator_name)
        )
    """)
    conn.commit()


def _download_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Download daily OHLCV from yfinance. Returns None on failure."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            logger.warning("yfinance returned empty data for %s", ticker)
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" not in df.columns and "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return None
    except Exception as e:
        logger.error("yfinance download failed for %s: %s", ticker, e)
        return None


def fetch_gold_silver(
    years: int = 5,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, int]:
    """
    Download XAU/USD and XAG/USD from yfinance and store in macro_indicators.

    Returns a dict with counts of rows inserted/updated per indicator.
    """
    end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(tz=timezone.utc) - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    logger.info("Fetching gold/silver prices (%s → %s)...", start_date, end_date)

    conn = sqlite3.connect(db_path, timeout=60)
    _ensure_macro_table(conn)
    counts: Dict[str, int] = {}

    # ---- Gold: GC=F --------------------------------------------------------
    gold_df = _download_yfinance("GC=F", start_date, end_date)
    if gold_df is not None and "close" in gold_df.columns:
        gold_df = gold_df[["date", "close"]].dropna().sort_values("date")
        gold_df["return"] = np.log(gold_df["close"] / gold_df["close"].shift(1))
        gold_df = gold_df.dropna(subset=["close"])

        n_price = _upsert_series(conn, gold_df, "date", "close", "gold_usd_per_oz", "yfinance:GC=F")
        n_ret = _upsert_series(conn, gold_df.dropna(subset=["return"]), "date", "return", "gold_return", "yfinance:GC=F")
        counts["gold_usd_per_oz"] = n_price
        counts["gold_return"] = n_ret
        logger.info("Gold: %d price rows, %d return rows inserted/updated", n_price, n_ret)
    else:
        logger.warning("Gold data unavailable from yfinance")

    # ---- Silver: SI=F ------------------------------------------------------
    silver_df = _download_yfinance("SI=F", start_date, end_date)
    if silver_df is not None and "close" in silver_df.columns:
        silver_df = silver_df[["date", "close"]].dropna().sort_values("date")
        silver_df["return"] = np.log(silver_df["close"] / silver_df["close"].shift(1))
        silver_df = silver_df.dropna(subset=["close"])

        n_price = _upsert_series(conn, silver_df, "date", "close", "silver_usd_per_oz", "yfinance:SI=F")
        n_ret = _upsert_series(conn, silver_df.dropna(subset=["return"]), "date", "return", "silver_return", "yfinance:SI=F")
        counts["silver_usd_per_oz"] = n_price
        counts["silver_return"] = n_ret
        logger.info("Silver: %d price rows, %d return rows inserted/updated", n_price, n_ret)
    else:
        logger.warning("Silver data unavailable from yfinance")

    # ---- USD/NPR: USDNPR=X -------------------------------------------------
    fx_df = _download_yfinance("USDNPR=X", start_date, end_date)
    if fx_df is not None and "close" in fx_df.columns and len(fx_df) > 10:
        fx_df = fx_df[["date", "close"]].dropna().sort_values("date")
        n_fx = _upsert_series(conn, fx_df, "date", "close", "usd_npr_rate", "yfinance:USDNPR=X")
        counts["usd_npr_rate"] = n_fx
        logger.info("USD/NPR: %d rows inserted/updated", n_fx)
    else:
        # Fallback: insert hardcoded approximations if not already present
        logger.info("USD/NPR forex unavailable from yfinance — inserting fallback approximations")
        n_fx = _insert_fallback_rates(conn)
        counts["usd_npr_rate"] = n_fx

    conn.commit()
    conn.close()
    return counts


def _upsert_series(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    indicator_name: str,
    source: str,
) -> int:
    """INSERT OR REPLACE rows into macro_indicators. Returns number of rows processed."""
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for _, row in df.iterrows():
        date_str = str(row[date_col])[:10]
        value = float(row[value_col])
        if np.isnan(value) or np.isinf(value):
            continue
        rows.append((date_str, indicator_name, value, source, now_utc))

    if not rows:
        return 0

    conn.executemany(
        """
        INSERT OR REPLACE INTO macro_indicators
            (date, indicator_name, value, source, scraped_at_utc)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _insert_fallback_rates(conn: sqlite3.Connection) -> int:
    """Insert hardcoded USD/NPR fallback rates (only if not already present)."""
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n = 0
    for date_str, rate in _FALLBACK_USDNPR:
        cur = conn.execute(
            "SELECT 1 FROM macro_indicators WHERE indicator_name='usd_npr_rate' AND date=?",
            (date_str,),
        )
        if cur.fetchone() is None:
            conn.execute(
                """INSERT INTO macro_indicators
                   (date, indicator_name, value, source, scraped_at_utc)
                   VALUES (?,?,?,?,?)""",
                (date_str, "usd_npr_rate", rate, "hardcoded_nrb_approx", now_utc),
            )
            n += 1
    return n


def get_gold_prices_df(
    db_path: str = DEFAULT_DB_PATH,
    as_of_date: Optional[str] = None,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """
    Retrieve gold price history from macro_indicators.

    Returns a DataFrame with columns:
        date, gold_usd, gold_return, usd_npr, gold_npr

    Rows are sorted chronologically, covering up to `lookback_days` before
    `as_of_date` (or latest available if as_of_date is None).
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30)

        ref_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.strptime(ref_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        # Gold prices
        gold_q = """
            SELECT date, value AS gold_usd FROM macro_indicators
            WHERE indicator_name = 'gold_usd_per_oz'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        gold_df = pd.read_sql_query(gold_q, conn, params=(start_date, ref_date))

        if gold_df.empty:
            conn.close()
            return pd.DataFrame(columns=["date", "gold_usd", "gold_return", "usd_npr", "gold_npr"])

        # Gold returns
        ret_q = """
            SELECT date, value AS gold_return FROM macro_indicators
            WHERE indicator_name = 'gold_return'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        ret_df = pd.read_sql_query(ret_q, conn, params=(start_date, ref_date))

        # USD/NPR rates
        fx_q = """
            SELECT date, value AS usd_npr FROM macro_indicators
            WHERE indicator_name = 'usd_npr_rate'
              AND date <= ?
            ORDER BY date
        """
        fx_df = pd.read_sql_query(fx_q, conn, params=(ref_date,))

        conn.close()

        # Merge
        merged = gold_df.merge(ret_df, on="date", how="left")

        # Forward-fill USD/NPR (annual approximations → need ffill to daily)
        if not fx_df.empty:
            fx_df = fx_df.sort_values("date")
            merged = merged.merge(fx_df, on="date", how="left")
            merged["usd_npr"] = merged["usd_npr"].ffill()
            # Fill remaining with last known fallback
            merged["usd_npr"] = merged["usd_npr"].fillna(
                _get_fallback_rate(ref_date)
            )
        else:
            merged["usd_npr"] = _get_fallback_rate(ref_date)

        merged["gold_npr"] = merged["gold_usd"] * merged["usd_npr"]
        merged = merged.sort_values("date").reset_index(drop=True)
        return merged

    except Exception as e:
        logger.warning("get_gold_prices_df failed: %s", e)
        return pd.DataFrame(columns=["date", "gold_usd", "gold_return", "usd_npr", "gold_npr"])


def get_silver_prices_df(
    db_path: str = DEFAULT_DB_PATH,
    as_of_date: Optional[str] = None,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """Retrieve silver price history from macro_indicators (same structure as gold)."""
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        ref_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.strptime(ref_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        silver_q = """
            SELECT date, value AS silver_usd FROM macro_indicators
            WHERE indicator_name = 'silver_usd_per_oz'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        silver_df = pd.read_sql_query(silver_q, conn, params=(start_date, ref_date))

        ret_q = """
            SELECT date, value AS silver_return FROM macro_indicators
            WHERE indicator_name = 'silver_return'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        ret_df = pd.read_sql_query(ret_q, conn, params=(start_date, ref_date))
        conn.close()

        if silver_df.empty:
            return pd.DataFrame(columns=["date", "silver_usd", "silver_return"])

        merged = silver_df.merge(ret_df, on="date", how="left")
        merged = merged.sort_values("date").reset_index(drop=True)
        return merged

    except Exception as e:
        logger.warning("get_silver_prices_df failed: %s", e)
        return pd.DataFrame(columns=["date", "silver_usd", "silver_return"])


# --------------------------------------------------------------------------
# Nepal FENEGOSIDA price persistence
# --------------------------------------------------------------------------

def store_nepal_metals_prices(
    gold_npr_per_tola: float,
    silver_npr_per_tola: float,
    date_str: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
    source: str = "FENEGOSIDA",
) -> bool:
    """
    Persist Nepal gold and silver prices (NPR/tola) from FENEGOSIDA into
    macro_indicators so CHG% can be computed from previous-day history.

    Indicator names stored:
        nepal_gold_npr_tola   — Fine gold, NPR per tola (FENEGOSIDA daily rate)
        nepal_silver_npr_tola — Silver, NPR per tola (FENEGOSIDA daily rate)

    Called automatically from the TUI each time prices are fetched.

    Returns True on success, False on failure.
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        _ensure_macro_table(conn)

        rows = []
        if gold_npr_per_tola and gold_npr_per_tola > 0:
            rows.append((date_str, "nepal_gold_npr_tola", float(gold_npr_per_tola), source, now_utc))
        if silver_npr_per_tola and silver_npr_per_tola > 0:
            rows.append((date_str, "nepal_silver_npr_tola", float(silver_npr_per_tola), source, now_utc))

        if rows:
            conn.executemany(
                """INSERT OR REPLACE INTO macro_indicators
                   (date, indicator_name, value, source, scraped_at_utc)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            logger.info(
                "Stored Nepal metals for %s: gold=NPR %s/tola, silver=NPR %s/tola",
                date_str, gold_npr_per_tola, silver_npr_per_tola,
            )
        conn.close()
        return True

    except Exception as e:
        logger.warning("store_nepal_metals_prices failed: %s", e)
        return False


def get_nepal_metals_history(
    db_path: str = DEFAULT_DB_PATH,
    lookback_days: int = 90,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retrieve Nepal gold and silver price history with day-over-day change %.

    Returns a DataFrame with columns:
        date, gold_npr_tola, silver_npr_tola,
        gold_chg_pct, silver_chg_pct,   ← day-over-day change
        gold_chg_abs, silver_chg_abs

    Rows sorted chronologically (oldest first).
    """
    ref_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.strptime(ref_date, "%Y-%m-%d") - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    try:
        conn = sqlite3.connect(db_path, timeout=30)

        gold_q = """
            SELECT date, value AS gold_npr_tola FROM macro_indicators
            WHERE indicator_name = 'nepal_gold_npr_tola'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        silver_q = """
            SELECT date, value AS silver_npr_tola FROM macro_indicators
            WHERE indicator_name = 'nepal_silver_npr_tola'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        gold_df = pd.read_sql_query(gold_q, conn, params=(start_date, ref_date))
        silver_df = pd.read_sql_query(silver_q, conn, params=(start_date, ref_date))
        conn.close()

        if gold_df.empty and silver_df.empty:
            return pd.DataFrame(columns=[
                "date", "gold_npr_tola", "silver_npr_tola",
                "gold_chg_pct", "silver_chg_pct", "gold_chg_abs", "silver_chg_abs",
            ])

        # Merge on date
        merged = gold_df.merge(silver_df, on="date", how="outer").sort_values("date")

        # Day-over-day change
        merged["gold_chg_abs"] = merged["gold_npr_tola"].diff()
        merged["gold_chg_pct"] = merged["gold_npr_tola"].pct_change() * 100

        merged["silver_chg_abs"] = merged["silver_npr_tola"].diff()
        merged["silver_chg_pct"] = merged["silver_npr_tola"].pct_change() * 100

        return merged.reset_index(drop=True)

    except Exception as e:
        logger.warning("get_nepal_metals_history failed: %s", e)
        return pd.DataFrame(columns=[
            "date", "gold_npr_tola", "silver_npr_tola",
            "gold_chg_pct", "silver_chg_pct", "gold_chg_abs", "silver_chg_abs",
        ])


def get_latest_nepal_metals(
    db_path: str = DEFAULT_DB_PATH,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Get latest Nepal gold/silver prices with CHG% from previous available day.

    Returns dict with keys:
        gold_npr_tola, silver_npr_tola,
        gold_chg_pct, silver_chg_pct,
        gold_chg_abs, silver_chg_abs,
        date, prev_date
    """
    ref_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
    df = get_nepal_metals_history(db_path=db_path, lookback_days=10, as_of_date=ref_date)

    empty = {
        "gold_npr_tola": None, "silver_npr_tola": None,
        "gold_chg_pct": None, "silver_chg_pct": None,
        "gold_chg_abs": None, "silver_chg_abs": None,
        "date": None, "prev_date": None,
    }

    if df.empty:
        return empty

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    return {
        "gold_npr_tola": latest.get("gold_npr_tola"),
        "silver_npr_tola": latest.get("silver_npr_tola"),
        "gold_chg_pct": round(float(latest["gold_chg_pct"]), 2)
            if pd.notna(latest.get("gold_chg_pct")) else None,
        "silver_chg_pct": round(float(latest["silver_chg_pct"]), 2)
            if pd.notna(latest.get("silver_chg_pct")) else None,
        "gold_chg_abs": round(float(latest["gold_chg_abs"]), 0)
            if pd.notna(latest.get("gold_chg_abs")) else None,
        "silver_chg_abs": round(float(latest["silver_chg_abs"]), 0)
            if pd.notna(latest.get("silver_chg_abs")) else None,
        "date": str(latest["date"]),
        "prev_date": str(prev["date"]) if prev is not None else None,
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fetch gold/silver prices from yfinance into macro_indicators"
    )
    parser.add_argument("--years", type=int, default=5, help="Years of history to fetch (default: 5)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite database")
    args = parser.parse_args()

    counts = fetch_gold_silver(years=args.years, db_path=args.db)
    print("\nIngestion complete:")
    for name, n in counts.items():
        print(f"  {name}: {n} rows")


if __name__ == "__main__":
    main()
