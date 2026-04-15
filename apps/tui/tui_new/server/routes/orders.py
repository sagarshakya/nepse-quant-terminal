"""Order management endpoints — paper trading order book."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.quant_pro.paths import ensure_dir, get_trading_runtime_dir

router = APIRouter()

ORDERS_FILE = ensure_dir(Path(get_trading_runtime_dir(__file__))) / "tui_paper_orders.json"
ORDER_HISTORY_FILE = ensure_dir(Path(get_trading_runtime_dir(__file__))) / "tui_paper_order_history.json"


def _load_orders(path: Path) -> list[dict]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _save_orders(path: Path, orders: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(orders, indent=2, default=str))


class OrderRequest(BaseModel):
    side: str
    symbol: str
    qty: int
    price: float
    slippage: float = 0.0


@router.post("")
async def submit_order(order: OrderRequest, request: Request):
    now = datetime.now()
    new_order = {
        "id": f"ORD-{uuid.uuid4().hex[:8]}",
        "side": order.side.upper(),
        "symbol": order.symbol.upper().strip(),
        "qty": order.qty,
        "price": order.price,
        "slippage": order.slippage,
        "status": "PENDING",
        "timestamp": now.isoformat(),
        "filled_price": None,
    }

    # Add to daily orders
    orders = _load_orders(ORDERS_FILE)
    orders.append(new_order)
    _save_orders(ORDERS_FILE, orders)

    # Also add to history
    history = _load_orders(ORDER_HISTORY_FILE)
    history.append(new_order)
    _save_orders(ORDER_HISTORY_FILE, history)

    # Try to execute via engine
    engine = request.app.state.engine
    if engine:
        try:
            if order.side.upper() == "BUY":
                from apps.classic.dashboard import exec_buy
                exec_buy(order.symbol.upper(), order.qty, order.price)
            elif order.side.upper() == "SELL":
                from apps.classic.dashboard import exec_sell
                exec_sell(order.symbol.upper(), order.qty, order.price)

            # Mark as filled
            new_order["status"] = "FILLED"
            new_order["filled_price"] = order.price
            _save_orders(ORDERS_FILE, orders)
            _save_orders(ORDER_HISTORY_FILE, history)
        except Exception as e:
            new_order["status"] = f"ERROR: {str(e)[:50]}"
            _save_orders(ORDERS_FILE, orders)
    else:
        # Paper mode: auto-fill
        new_order["status"] = "FILLED"
        new_order["filled_price"] = order.price * (1 + order.slippage / 100 if order.side.upper() == "BUY" else 1 - order.slippage / 100)
        _save_orders(ORDERS_FILE, orders)
        _save_orders(ORDER_HISTORY_FILE, history)

    return {"status": new_order["status"], "order_id": new_order["id"]}


@router.get("/daily")
async def daily_orders(request: Request):
    today = datetime.now().strftime("%Y-%m-%d")
    orders = _load_orders(ORDERS_FILE)
    # Filter to today's orders
    return [o for o in orders if o.get("timestamp", "").startswith(today)]


@router.get("/history")
async def order_history(request: Request):
    return _load_orders(ORDER_HISTORY_FILE)[-100:]  # Last 100
