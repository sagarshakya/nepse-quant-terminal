"""Portfolio endpoints — wraps TUITradingEngine."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
async def get_portfolio(request: Request):
    engine = request.app.state.engine
    if not engine:
        return {
            "positions": [],
            "cash": 1_000_000,
            "nav": 1_000_000,
            "total_cost": 0,
            "total_value": 0,
            "total_return": 0,
            "day_pnl": 0,
            "day_ret": 0,
            "realized": 0,
            "unrealized": 0,
            "max_dd": 0,
            "regime": "unknown",
            "engine_phase": "idle",
            "sector_exposure": {},
        }

    try:
        stats = engine.get_portfolio_stats()
        positions = []
        for p in stats.get("positions", []):
            positions.append({
                "symbol": p.get("symbol", ""),
                "shares": p.get("shares", 0),
                "entry_price": p.get("entry_price", 0),
                "entry_date": p.get("entry_date", ""),
                "ltp": p.get("ltp", 0),
                "cost_basis": p.get("cost_basis", 0),
                "market_value": p.get("market_value", 0),
                "unrealized_pnl": p.get("unrealized_pnl", 0),
                "pnl_pct": p.get("pnl_pct", 0),
                "weight": p.get("weight", 0),
                "signal_type": p.get("signal_type", ""),
                "holding_days": p.get("holding_days", 0),
            })
        return {
            "positions": positions,
            "cash": stats.get("cash", 0),
            "nav": stats.get("nav", 0),
            "total_cost": stats.get("total_cost", 0),
            "total_value": stats.get("total_value", 0),
            "total_return": stats.get("total_return", 0),
            "day_pnl": stats.get("day_pnl", 0),
            "day_ret": stats.get("day_ret", 0),
            "realized": stats.get("realized", 0),
            "unrealized": stats.get("unrealized", 0),
            "max_dd": stats.get("max_dd", 0),
            "regime": stats.get("regime", "unknown"),
            "engine_phase": stats.get("engine_phase", "idle"),
            "sector_exposure": stats.get("sector_exposure", {}),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/trades")
async def get_trades(request: Request):
    engine = request.app.state.engine
    if not engine:
        return []
    try:
        df = engine.get_trade_log()
        if df is None or df.empty:
            return []
        records = df.to_dict(orient="records")
        return [
            {
                "date": str(r.get("Date", r.get("date", ""))),
                "action": str(r.get("Action", r.get("action", ""))),
                "symbol": str(r.get("Symbol", r.get("symbol", ""))),
                "shares": int(r.get("Shares", r.get("shares", 0))),
                "price": float(r.get("Price", r.get("price", 0))),
                "fees": float(r.get("Fees", r.get("fees", 0))),
                "reason": str(r.get("Reason", r.get("reason", ""))),
                "pnl": float(r.get("PnL", r.get("pnl", 0))),
                "pnl_pct": float(r.get("PnL_Pct", r.get("pnl_pct", 0))),
            }
            for r in records
        ]
    except Exception:
        return []
