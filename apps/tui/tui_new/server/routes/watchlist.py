"""Watchlist endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.quant_pro.paths import get_trading_runtime_dir

router = APIRouter()

WATCHLIST_FILE = Path(get_trading_runtime_dir(__file__)) / "watchlist.json"
DEFAULT_WATCHLIST = [
    "NABIL", "NLIC", "UPPER", "CHDC", "SBL", "SHIVM", "NRIC",
    "NTC", "NICA", "GBIME", "KBL", "MEGA", "PRVU", "SBI",
]


def _load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        try:
            data = json.loads(WATCHLIST_FILE.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return DEFAULT_WATCHLIST[:]


def _save_watchlist(symbols: list[str]):
    WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_FILE.write_text(json.dumps(symbols, indent=2))


@router.get("")
async def get_watchlist(request: Request):
    symbols = _load_watchlist()
    md = request.app.state.md
    items = []
    for sym in symbols:
        item = {"symbol": sym, "ltp": 0, "change": 0, "change_pct": 0, "volume": 0}
        if hasattr(md, 'df') and not md.df.empty:
            row = md.df[md.df["symbol"] == sym]
            if not row.empty:
                r = row.iloc[0]
                item["ltp"] = float(r.get("ltp", r.get("close", 0)))
                item["change_pct"] = float(r.get("pc", r.get("chg_pct", 0)))
                item["volume"] = int(r.get("vol", r.get("volume", 0)))
        items.append(item)
    return items


class WatchlistAction(BaseModel):
    action: str  # "add" or "remove"
    symbol: str


@router.post("")
async def update_watchlist(body: WatchlistAction, request: Request):
    symbols = _load_watchlist()
    if body.action == "add" and body.symbol not in symbols:
        symbols.append(body.symbol)
    elif body.action == "remove" and body.symbol in symbols:
        symbols.remove(body.symbol)
    _save_watchlist(symbols)
    return {"status": "ok", "watchlist": symbols}
