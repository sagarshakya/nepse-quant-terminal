"""MarketService — read-only view over the existing NEPSE SQLite store.

No writes. No side-effects on the database the TUI/batch jobs depend on.
All reads go through a short-lived sqlite3 connection in read-only mode where
possible, so the service can't corrupt the TUI's runtime state.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from backend.quant_pro.database import get_db_path
from backend.core.types import Quote, OHLCV, MarketSnapshot, IndexPoint


# Symbols treated as index / header strip. NEPSE is the main benchmark; rest
# appear only if present in the database.
INDEX_SYMBOLS: tuple[str, ...] = (
    "NEPSE", "BANKING", "DEVELOPMENT BANK", "HYDROPOWER", "FINANCE",
    "MICROFINANCE", "LIFE INSURANCE", "NON LIFE INSURANCE", "HOTELS",
    "MANUFACTURING", "TRADING", "INVESTMENT", "OTHERS", "SENSITIVE",
)


class MarketService:
    """Thin wrapper over `nepse_market_data.db`. Connection-per-call.

    The service caches per-symbol OHLCV arrays in-process (LRU by recency).
    Nothing is mutated on disk.
    """

    def __init__(self, db_path: Optional[Path] = None, cache_size: int = 512):
        self._db_path = Path(db_path) if db_path is not None else Path(get_db_path())
        self._cache_size = cache_size
        self._ohlcv_cache: dict[str, tuple[float, OHLCV]] = {}
        self._snapshot_cache: dict[tuple, tuple[float, MarketSnapshot]] = {}
        self._snapshot_ttl = 15.0  # seconds

    # ---------- connections -----------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # read-only URI so the GUI can never corrupt the file underneath the TUI
        uri = f"file:{self._db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------- universe --------------------------------------------------
    def symbols(self, *, include_indices: bool = False) -> list[str]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol"
            )
            all_syms = [r[0] for r in cur.fetchall()]
        if include_indices:
            return all_syms
        idx = set(INDEX_SYMBOLS)
        return [s for s in all_syms if s not in idx]

    def last_trading_day(self) -> Optional[date]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT MAX(date) FROM stock_prices WHERE symbol = 'NEPSE'"
            )
            row = cur.fetchone()
        if not row or not row[0]:
            return None
        return datetime.strptime(row[0], "%Y-%m-%d").date()

    # ---------- snapshot --------------------------------------------------
    def snapshot(self, symbols: Optional[list[str]] = None) -> MarketSnapshot:
        """Latest EOD snapshot for the requested symbols (or all non-index symbols).

        Uses `stock_prices` — the same source the TUI reads from — and joins
        last/prev close per symbol for change / change_pct.
        """
        if symbols is None:
            symbols = self.symbols(include_indices=False)

        if not symbols:
            return MarketSnapshot(as_of=datetime.now(), quotes=[])

        cache_key = tuple(symbols)
        now = time.monotonic()
        cached = self._snapshot_cache.get(cache_key)
        if cached and (now - cached[0]) < self._snapshot_ttl:
            return cached[1]

        placeholders = ",".join(["?"] * len(symbols))
        # Fast path: restrict to rows within the last ~20 calendar days, then
        # take the two most recent per symbol. The index on (symbol, date)
        # makes this a narrow range scan per symbol.
        q = f"""
        WITH recent AS (
            SELECT symbol, date, open, high, low, close, volume
            FROM stock_prices
            WHERE symbol IN ({placeholders})
              AND date >= date((SELECT MAX(date) FROM stock_prices WHERE symbol='NEPSE'), '-30 days')
        ),
        ranked AS (
            SELECT symbol, date, open, high, low, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM recent
        )
        SELECT symbol, date, open, high, low, close, volume, rn
        FROM ranked
        WHERE rn <= 2
        ORDER BY symbol, rn
        """
        with self._connect() as conn:
            df = pd.read_sql_query(q, conn, params=symbols)

        quotes: list[Quote] = []
        as_of = datetime.now()
        advancers = decliners = unchanged = 0

        for sym, grp in df.groupby("symbol", sort=False):
            grp = grp.sort_values("rn")
            last = grp.iloc[0]
            prev_close = float(grp.iloc[1]["close"]) if len(grp) > 1 else float(last["close"])
            close = float(last["close"])
            change = close - prev_close
            change_pct = (change / prev_close * 100.0) if prev_close else 0.0
            quotes.append(Quote(
                symbol=sym,
                last=close,
                change=change,
                change_pct=change_pct,
                volume=float(last["volume"]),
                high=float(last["high"]),
                low=float(last["low"]),
                open=float(last["open"]),
                prev_close=prev_close,
                as_of=datetime.strptime(last["date"], "%Y-%m-%d"),
            ))
            if change > 0:
                advancers += 1
            elif change < 0:
                decliners += 1
            else:
                unchanged += 1

        # NEPSE index change % if available
        idx_change_pct = 0.0
        try:
            idx_q = self._single_quote("NEPSE")
            if idx_q is not None:
                idx_change_pct = idx_q.change_pct
        except Exception:
            pass

        quotes.sort(key=lambda q: q.symbol)
        snap = MarketSnapshot(
            as_of=as_of,
            quotes=quotes,
            index_change_pct=idx_change_pct,
            advancers=advancers,
            decliners=decliners,
            unchanged=unchanged,
        )
        self._snapshot_cache[cache_key] = (now, snap)
        if len(self._snapshot_cache) > 64:
            oldest = sorted(self._snapshot_cache.items(), key=lambda kv: kv[1][0])
            for key, _ in oldest[: len(oldest) // 4]:
                self._snapshot_cache.pop(key, None)
        return snap

    def _single_quote(self, symbol: str) -> Optional[Quote]:
        snap = self.snapshot([symbol])
        return snap.quotes[0] if snap.quotes else None

    def index_strip(self) -> list[IndexPoint]:
        """Indices for the always-on top strip. One round-trip against
        benchmark_index_history (the table the scrapers populate)."""
        sql = """
        WITH recent AS (
            SELECT benchmark, date, close
            FROM benchmark_index_history
            WHERE date >= date((SELECT MAX(date) FROM benchmark_index_history), '-30 days')
        ),
        ranked AS (
            SELECT benchmark, date, close,
                   ROW_NUMBER() OVER (PARTITION BY benchmark ORDER BY date DESC) AS rn
            FROM recent
        )
        SELECT benchmark, close, rn FROM ranked WHERE rn <= 2 ORDER BY benchmark, rn
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        by_sym: dict[str, list[float]] = {}
        for r in rows:
            by_sym.setdefault(r["benchmark"], []).append(float(r["close"]))
        # Preserve preferred ordering from INDEX_SYMBOLS
        order = {s: i for i, s in enumerate(INDEX_SYMBOLS)}
        out: list[IndexPoint] = []
        for sym in sorted(by_sym.keys(), key=lambda s: order.get(s, 999)):
            vals = by_sym[sym]
            last = vals[0]
            prev = vals[1] if len(vals) > 1 else last
            pct = ((last - prev) / prev * 100.0) if prev else 0.0
            out.append(IndexPoint(symbol=sym, last=last, change_pct=pct))
        return out

    # ---------- history ---------------------------------------------------
    def history(
        self,
        symbol: str,
        *,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> OHLCV:
        """Daily OHLCV history for a symbol, optionally date-windowed."""
        cached = self._ohlcv_cache.get(symbol)
        now = time.monotonic()
        if cached and (now - cached[0]) < 60.0 and start is None and end is None:
            return cached[1]

        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM stock_prices "
                "WHERE symbol = ? ORDER BY date ASC",
                conn,
                params=(symbol,),
            )

        if df.empty:
            ohlcv = OHLCV(
                symbol=symbol,
                dates=np.array([], dtype="datetime64[D]"),
                open=np.array([], dtype=np.float64),
                high=np.array([], dtype=np.float64),
                low=np.array([], dtype=np.float64),
                close=np.array([], dtype=np.float64),
                volume=np.array([], dtype=np.float64),
            )
            return ohlcv

        dates = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
        if start is not None:
            mask = dates >= np.datetime64(start)
            df = df.loc[mask].reset_index(drop=True)
            dates = dates[mask]
        if end is not None:
            mask = dates <= np.datetime64(end)
            df = df.loc[mask].reset_index(drop=True)
            dates = dates[mask]

        ohlcv = OHLCV(
            symbol=symbol,
            dates=dates,
            open=df["open"].to_numpy(dtype=np.float64),
            high=df["high"].to_numpy(dtype=np.float64),
            low=df["low"].to_numpy(dtype=np.float64),
            close=df["close"].to_numpy(dtype=np.float64),
            volume=df["volume"].to_numpy(dtype=np.float64),
        )
        if start is None and end is None:
            self._ohlcv_cache[symbol] = (now, ohlcv)
            # simple LRU trim
            if len(self._ohlcv_cache) > self._cache_size:
                oldest = sorted(self._ohlcv_cache.items(), key=lambda kv: kv[1][0])
                for key, _ in oldest[: len(oldest) // 4]:
                    self._ohlcv_cache.pop(key, None)
        return ohlcv

    # ---------- intraday --------------------------------------------------
    def intraday(
        self,
        symbol: str,
        *,
        bucket_minutes: int = 5,
        preferred_session_date: Optional[str] = None,
    ) -> OHLCV:
        """Intraday OHLCV bars aggregated from stored market_quotes snapshots.

        Bar timestamps are bucket starts in NST (UTC+5:45) as datetime64[m].
        Returns empty OHLCV if the market_quotes table is absent or has no rows
        for this symbol.
        """
        sym = str(symbol or "").strip().upper()
        empty = OHLCV(
            symbol=sym,
            dates=np.array([], dtype="datetime64[m]"),
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.float64),
        )
        if not sym:
            return empty

        try:
            with self._connect() as conn:
                target_date = preferred_session_date
                if target_date:
                    row = conn.execute(
                        """
                        SELECT COUNT(*) AS cnt FROM market_quotes
                        WHERE symbol = ?
                          AND date(datetime(fetched_at_utc, '+5 hours', '+45 minutes')) = ?
                        """,
                        (sym, target_date),
                    ).fetchone()
                    if not row or int(row[0] or 0) == 0:
                        target_date = None

                if not target_date:
                    row = conn.execute(
                        """
                        SELECT MAX(date(datetime(fetched_at_utc, '+5 hours', '+45 minutes'))) AS d
                        FROM market_quotes WHERE symbol = ?
                        """,
                        (sym,),
                    ).fetchone()
                    target_date = row[0] if row else None

                if not target_date:
                    return empty

                df = pd.read_sql_query(
                    """
                    SELECT fetched_at_utc, last_traded_price, close_price, total_trade_quantity
                    FROM market_quotes
                    WHERE symbol = ?
                      AND date(datetime(fetched_at_utc, '+5 hours', '+45 minutes')) = ?
                    ORDER BY fetched_at_utc ASC
                    """,
                    conn, params=(sym, target_date),
                )
        except sqlite3.Error:
            return empty

        if df.empty:
            return empty

        df["price"] = pd.to_numeric(df["last_traded_price"], errors="coerce")
        df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
        df["price"] = df["price"].fillna(df["close_price"])
        df = df[df["price"] > 0].copy()
        if df.empty:
            return empty

        df["cum_qty"] = pd.to_numeric(df["total_trade_quantity"], errors="coerce").ffill().fillna(0.0)
        df["volume"] = df["cum_qty"].diff().clip(lower=0).fillna(0.0)
        df["ts_nst"] = pd.to_datetime(df["fetched_at_utc"], utc=True) + pd.Timedelta(hours=5, minutes=45)
        bm = max(1, int(bucket_minutes))
        df["bucket"] = df["ts_nst"].dt.floor(f"{bm}min").dt.tz_localize(None)

        bars = (
            df.groupby("bucket", sort=True)
              .agg(open=("price", "first"), high=("price", "max"),
                   low=("price", "min"), close=("price", "last"),
                   volume=("volume", "sum"))
              .reset_index()
        )

        if len(bars) < 2 and len(df) >= 2:
            bars = pd.DataFrame({
                "bucket": df["ts_nst"].dt.tz_localize(None),
                "open": df["price"], "high": df["price"],
                "low": df["price"], "close": df["price"],
                "volume": df["volume"],
            })

        dates = bars["bucket"].values.astype("datetime64[m]")
        return OHLCV(
            symbol=sym, dates=dates,
            open=bars["open"].to_numpy(dtype=np.float64),
            high=bars["high"].to_numpy(dtype=np.float64),
            low=bars["low"].to_numpy(dtype=np.float64),
            close=bars["close"].to_numpy(dtype=np.float64),
            volume=bars["volume"].to_numpy(dtype=np.float64),
        )

    # ---------- sector map / corporate actions ----------------------------
    _SECTOR_ALIASES = {
        "hydro power": "Hydropower",
        "hydropower": "Hydropower",
        "commercial bank": "Commercial Banks",
        "commercial banks": "Commercial Banks",
        "development bank": "Development Banks",
        "development banks": "Development Banks",
        "microfinance": "Microfinance",
        "finance": "Finance",
        "life insurance": "Life Insurance",
        "non-life insurance": "Non-Life Insurance",
        "non life insurance": "Non-Life Insurance",
        "mutual fund": "Mutual Fund",
        "investment": "Investment",
        "hotel & tourism": "Hotels & Tourism",
        "hotels and tourism": "Hotels & Tourism",
        "manufacturing and processing": "Manufacturing & Processing",
        "manufacturing & processing": "Manufacturing & Processing",
        "tradings": "Trading",
        "trading": "Trading",
        "others": "Others",
    }

    def sector_map(self) -> dict[str, str]:
        """Return {symbol: canonical sector name} from latest fundamentals row."""
        sql = """
        WITH latest AS (
            SELECT symbol, sector,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM fundamentals
            WHERE sector IS NOT NULL AND sector != ''
        )
        SELECT symbol, sector FROM latest WHERE rn = 1
        """
        out: dict[str, str] = {}
        try:
            with self._connect() as conn:
                for r in conn.execute(sql).fetchall():
                    raw = (r["sector"] or "").strip()
                    key = raw.lower()
                    out[r["symbol"]] = self._SECTOR_ALIASES.get(key, raw or "Others")
        except sqlite3.Error:
            return {}
        return out

    def upcoming_corporate_actions(self, *, days: int = 30, limit: int = 40) -> list[dict]:
        """Bookclose events coming up in the next N days."""
        sql = """
            SELECT symbol, bookclose_date, cash_dividend_pct, bonus_share_pct,
                   right_share_ratio, description
            FROM corporate_actions
            WHERE bookclose_date IS NOT NULL
              AND bookclose_date >= date('now')
              AND bookclose_date <= date('now', ?)
            ORDER BY bookclose_date ASC
            LIMIT ?
        """
        out: list[dict] = []
        try:
            with self._connect() as conn:
                for r in conn.execute(sql, (f"+{days} days", limit)).fetchall():
                    out.append({
                        "symbol":       r["symbol"],
                        "book_close":   r["bookclose_date"] or "",
                        "cash":         float(r["cash_dividend_pct"] or 0.0),
                        "bonus":        float(r["bonus_share_pct"] or 0.0),
                        "right":        r["right_share_ratio"] or "",
                        "description":  r["description"] or "",
                    })
        except sqlite3.Error:
            return []
        return out

    def corporate_actions_for_symbol(self, symbol: str, *, limit: int = 10) -> list[dict]:
        """Corporate actions history for a specific symbol (most recent first)."""
        sql = """
            SELECT fiscal_year, bookclose_date, cash_dividend_pct, bonus_share_pct,
                   right_share_ratio, description
            FROM corporate_actions
            WHERE symbol = ?
            ORDER BY bookclose_date DESC
            LIMIT ?
        """
        out: list[dict] = []
        try:
            with self._connect() as conn:
                for r in conn.execute(sql, (symbol, limit)).fetchall():
                    out.append({
                        "fiscal_year": r["fiscal_year"] or "",
                        "book_close":  r["bookclose_date"] or "",
                        "cash":        float(r["cash_dividend_pct"] or 0.0),
                        "bonus":       float(r["bonus_share_pct"] or 0.0),
                        "right":       r["right_share_ratio"] or "",
                        "description": r["description"] or "",
                    })
        except sqlite3.Error:
            return []
        return out

    def quarterly_earnings(self, symbol: str, *, limit: int = 8) -> list[dict]:
        """Quarterly earnings rows for a symbol (most recent first)."""
        sql = """
            SELECT fiscal_year, quarter, eps, net_profit, revenue, book_value,
                   announcement_date
            FROM quarterly_earnings
            WHERE symbol = ?
            ORDER BY fiscal_year DESC, quarter DESC
            LIMIT ?
        """
        out: list[dict] = []
        try:
            with self._connect() as conn:
                for r in conn.execute(sql, (symbol, limit)).fetchall():
                    out.append({
                        "fiscal_year":  r["fiscal_year"] or "",
                        "quarter":      int(r["quarter"] or 0),
                        "eps":          r["eps"],
                        "net_profit":   r["net_profit"],
                        "revenue":      r["revenue"],
                        "book_value":   r["book_value"],
                        "announcement": r["announcement_date"] or "",
                    })
        except sqlite3.Error:
            return []
        return out

    def fundamentals_latest(self, symbol: str) -> Optional[dict]:
        """Most recent fundamentals row for a symbol (PE/PB/EPS/BVPS/market cap…)."""
        sql = """
            SELECT date, market_cap, pe_ratio, pb_ratio, eps, book_value_per_share,
                   roe, dividend_yield, shares_outstanding, sector
            FROM fundamentals
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 1
        """
        try:
            with self._connect() as conn:
                r = conn.execute(sql, (symbol,)).fetchone()
        except sqlite3.Error:
            return None
        if not r:
            return None
        return {
            "date":         r["date"] or "",
            "market_cap":   r["market_cap"],
            "pe":           r["pe_ratio"],
            "pb":           r["pb_ratio"],
            "eps":          r["eps"],
            "bvps":         r["book_value_per_share"],
            "roe":          r["roe"],
            "div_yield":    r["dividend_yield"],
            "shares_out":   r["shares_outstanding"],
            "sector":       r["sector"] or "",
        }

    # ---------- news -------------------------------------------------------
    def recent_news(self, *, limit: int = 60) -> list[dict]:
        """Recent news headlines (best-effort; returns [] if table missing)."""
        sql = """
            SELECT symbol, date, headline, source, sentiment_label, sentiment_score
            FROM news
            WHERE headline IS NOT NULL AND headline != ''
            ORDER BY COALESCE(date, created_at) DESC, id DESC
            LIMIT ?
        """
        out: list[dict] = []
        try:
            with self._connect() as conn:
                for r in conn.execute(sql, (limit,)).fetchall():
                    out.append({
                        "symbol":    r["symbol"] or "",
                        "date":      r["date"] or "",
                        "headline":  r["headline"] or "",
                        "source":    r["source"] or "",
                        "sentiment": r["sentiment_label"] or "",
                        "score":     float(r["sentiment_score"] or 0.0),
                    })
        except sqlite3.Error:
            return []
        return out

    def top_movers(self, n: int = 5) -> tuple[list[Quote], list[Quote], list[Quote]]:
        """(top gainers, top losers, top volume) from the latest snapshot."""
        snap = self.snapshot()
        quotes = snap.quotes
        gainers = sorted(quotes, key=lambda q: q.change_pct, reverse=True)[:n]
        losers  = sorted(quotes, key=lambda q: q.change_pct)[:n]
        vol_top = sorted(quotes, key=lambda q: q.volume, reverse=True)[:n]
        return gainers, losers, vol_top

    # ---------- breadth / sector -----------------------------------------
    def sector_breadth(self) -> dict[str, dict]:
        """Advancers/decliners by (coarse) sector name from sector indices.

        Cheap implementation — returns the index-level change % per SECTOR_INDEX
        symbol. Good enough for the Market Overview pane until the real sector
        mapping is lifted from quant_pro.data_io.
        """
        out: dict[str, dict] = {}
        with self._connect() as conn:
            for sym in INDEX_SYMBOLS:
                cur = conn.execute(
                    "SELECT date, close FROM stock_prices "
                    "WHERE symbol = ? ORDER BY date DESC LIMIT 2",
                    (sym,),
                )
                rows = cur.fetchall()
                if len(rows) == 2:
                    last, prev = float(rows[0]["close"]), float(rows[1]["close"])
                    out[sym] = {
                        "last": last,
                        "change_pct": (last - prev) / prev * 100.0 if prev else 0.0,
                        "as_of": rows[0]["date"],
                    }
        return out
