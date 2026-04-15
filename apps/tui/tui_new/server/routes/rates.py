"""Rates endpoints — forex, metals, energy, kalimati.

Directly imports rate-fetching functions from the existing TUI codebase.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter

# Ensure project root is importable for the rate-fetching functions
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()

# Cache to avoid hitting external APIs on every request
_cache: dict = {"data": None, "ts": None}
CACHE_TTL_S = 120  # 2 minutes


def _is_cache_fresh() -> bool:
    if _cache["ts"] is None or _cache["data"] is None:
        return False
    return (datetime.utcnow() - _cache["ts"]).total_seconds() < CACHE_TTL_S


def _fetch_all_rates() -> dict:
    """Fetch all rates from external APIs, with caching."""
    if _is_cache_fresh():
        return _cache["data"]

    result: dict = {"forex": [], "metals": [], "energy": [], "kalimati": []}

    # Forex rates from NRB
    try:
        from apps.tui.dashboard_tui import _fetch_nrb_forex_rates
        forex_raw = _fetch_nrb_forex_rates(("USD", "EUR", "GBP", "INR", "CNY", "JPY"))
        result["forex"] = [
            {
                "currency": r.get("currency_code", ""),
                "buy": r.get("buy_rate", 0),
                "sell": r.get("sell_rate", 0),
                "unit": r.get("unit", 1),
            }
            for r in forex_raw
        ]
    except Exception:
        pass

    # Gold & Silver from FENEGOSIDA
    try:
        from apps.tui.dashboard_tui import _fetch_gold_silver_prices
        metals = _fetch_gold_silver_prices()
        if metals:
            result["metals"] = [
                {"name": "Gold (per tola)", "price": metals.get("gold_per_tola", 0), "change": 0},
                {"name": "Silver (per tola)", "price": metals.get("silver_per_tola", 0), "change": 0},
            ]
    except Exception:
        pass

    # Crude oil from Yahoo
    try:
        from apps.tui.dashboard_tui import _fetch_yahoo_futures_price
        crude = _fetch_yahoo_futures_price("CL=F", "Crude Oil")
        if crude:
            result["energy"].append({
                "name": crude.get("label", "Crude Oil"),
                "price": crude.get("value", 0),
                "unit": crude.get("unit", "USD"),
            })
    except Exception:
        pass

    # NOC fuel prices
    try:
        from apps.tui.dashboard_tui import _fetch_noc_fuel_prices
        noc = _fetch_noc_fuel_prices()
        if noc:
            for fuel_key in ("petrol", "diesel", "kerosene", "lpg"):
                price = noc.get(fuel_key, 0)
                if price > 0:
                    result["energy"].append({
                        "name": fuel_key.capitalize(),
                        "price": price,
                        "unit": "NPR/L" if fuel_key != "lpg" else "NPR/cyl",
                    })
    except Exception:
        pass

    # Kalimati market
    try:
        from backend.market.kalimati_market import init_kalimati_db, get_kalimati_display_rows
        init_kalimati_db()
        kalimati_rows = get_kalimati_display_rows()
        result["kalimati"] = [
            {
                "name": r.get("commodity", r.get("name", "")),
                "unit": r.get("unit", "kg"),
                "min_price": float(r.get("min", r.get("min_price", 0))),
                "max_price": float(r.get("max", r.get("max_price", 0))),
                "avg_price": float(r.get("avg", r.get("avg_price", 0))),
            }
            for r in (kalimati_rows or [])
        ]
    except Exception:
        pass

    _cache["data"] = result
    _cache["ts"] = datetime.utcnow()
    return result


@router.get("")
async def get_rates():
    return _fetch_all_rates()
