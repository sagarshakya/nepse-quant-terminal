"""Lookup endpoints — symbol detail, chart data, financials, report."""
from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Request, Query

from backend.quant_pro.database import get_db_path

router = APIRouter()


@router.get("/{symbol}")
async def get_lookup(symbol: str, request: Request):
    md = request.app.state.md
    try:
        md.refresh()
    except Exception:
        pass

    detail = {
        "symbol": symbol,
        "name": "",
        "sector": "",
        "ltp": 0,
        "change": 0,
        "change_pct": 0,
        "pe_ratio": None,
        "market_cap": None,
        "high_52w": None,
        "low_52w": None,
        "eps": None,
        "book_value": None,
    }

    # Get data from MD
    if hasattr(md, 'df') and not md.df.empty:
        row = md.df[md.df["symbol"] == symbol]
        if not row.empty:
            r = row.iloc[0]
            detail["ltp"] = float(r.get("ltp", r.get("close", 0)))
            detail["change_pct"] = float(r.get("pc", r.get("chg_pct", 0)))
            detail["change"] = float(r.get("chg", 0))
            if "high_52w" in r:
                detail["high_52w"] = float(r["high_52w"]) if pd.notna(r["high_52w"]) else None
            if "low_52w" in r:
                detail["low_52w"] = float(r["low_52w"]) if pd.notna(r["low_52w"]) else None

    # OHLCV data
    ohlcv = []
    try:
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM stock_prices "
            "WHERE symbol=? ORDER BY date DESC LIMIT 250",
            conn,
            params=(symbol,),
        )
        conn.close()
        if not df.empty:
            df = df.sort_values("date")
            ohlcv = [
                {
                    "date": str(r["date"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": int(r["volume"]),
                }
                for _, r in df.iterrows()
            ]
    except Exception:
        pass

    # Financials
    financials = []
    # TODO: pull from quarterly_reports table if exists

    # Corporate actions
    corp_actions = []
    try:
        conn = sqlite3.connect(get_db_path())
        ca_df = pd.read_sql_query(
            "SELECT fiscal_year, bookclose_date, cash_dividend_pct, bonus_share_pct, "
            "right_share_ratio, agenda FROM corporate_actions "
            "WHERE symbol=? ORDER BY bookclose_date DESC LIMIT 10",
            conn,
            params=(symbol,),
        )
        conn.close()
        if not ca_df.empty:
            corp_actions = [
                {
                    "symbol": symbol,
                    "fiscal_year": str(r.get("fiscal_year", "")),
                    "bookclose_date": str(r.get("bookclose_date", "")),
                    "cash_dividend_pct": float(r.get("cash_dividend_pct", 0) or 0),
                    "bonus_share_pct": float(r.get("bonus_share_pct", 0) or 0),
                    "right_share_ratio": str(r.get("right_share_ratio", "") or ""),
                    "agenda": str(r.get("agenda", "") or ""),
                }
                for _, r in ca_df.iterrows()
            ]
    except Exception:
        pass

    # Report
    report = ""
    try:
        from backend.quant_pro.stock_report import build_stock_report
        report = build_stock_report(symbol) or ""
    except Exception:
        pass

    return {
        "detail": detail,
        "ohlcv": ohlcv,
        "financials": financials,
        "corporate_actions": corp_actions,
        "report": report,
    }


@router.get("/{symbol}/chart")
async def get_chart(symbol: str, tf: str = Query(default="D")):
    try:
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM stock_prices "
            "WHERE symbol=? ORDER BY date DESC LIMIT 500",
            conn,
            params=(symbol,),
        )
        conn.close()
        if df.empty:
            return []

        df = df.sort_values("date")

        # Resample if needed
        if tf == "W":
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").resample("W").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna().reset_index()
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        elif tf == "M":
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").resample("ME").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna().reset_index()
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        return [
            {
                "date": str(r["date"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r["volume"]),
            }
            for _, r in df.iterrows()
        ]
    except Exception:
        return []


@router.get("/{symbol}/report")
async def get_report(symbol: str):
    try:
        from backend.quant_pro.stock_report import build_stock_report
        report = build_stock_report(symbol)
        return {"report": report or f"No report available for {symbol}"}
    except Exception as e:
        return {"report": f"Report unavailable: {str(e)}"}
