"""Strategy configuration and backtesting endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
async def get_strategies(request: Request):
    # TODO: integrate with strategy registry
    try:
        from configs.long_term import LONG_TERM_CONFIG
        from configs.short_term import SHORT_TERM_CONFIG
        return [
            {
                "name": "Long-Term (C31)",
                "description": "Main quantitative strategy with regime-dependent sector limits",
                "signal_types": LONG_TERM_CONFIG.get("signal_types", []),
                "holding_days": LONG_TERM_CONFIG.get("holding_days", 40),
                "max_positions": LONG_TERM_CONFIG.get("max_positions", 5),
                "stop_loss_pct": LONG_TERM_CONFIG.get("stop_loss_pct", 8),
                "trailing_stop_pct": LONG_TERM_CONFIG.get("trailing_stop_pct", 10),
                "sector_limit": LONG_TERM_CONFIG.get("sector_limit", 0.5),
            },
            {
                "name": "Short-Term (Dividend)",
                "description": "Event-driven corporate action strategy",
                "signal_types": SHORT_TERM_CONFIG.get("signal_types", []),
                "holding_days": SHORT_TERM_CONFIG.get("holding_days", 12),
                "max_positions": SHORT_TERM_CONFIG.get("max_positions", 3),
                "stop_loss_pct": SHORT_TERM_CONFIG.get("stop_loss_pct", 6),
                "trailing_stop_pct": SHORT_TERM_CONFIG.get("trailing_stop_pct", 0),
                "sector_limit": SHORT_TERM_CONFIG.get("sector_limit", 1.0),
            },
        ]
    except Exception:
        return []


@router.post("/backtest")
async def run_backtest(request: Request):
    # TODO: integrate with backtesting engine
    return {
        "total_return": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "win_rate": 0,
        "total_trades": 0,
        "start_date": "",
        "end_date": "",
    }
