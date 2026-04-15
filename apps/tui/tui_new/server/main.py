"""
FastAPI bridge server — thin REST layer wrapping existing Python backend.
Run: python3 -m uvicorn apps.tui.tui_new.server.main:app --port 8100
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # apps/tui/tui_new/server -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from apps.tui.tui_new.server.routes import market, portfolio, signals, lookup, orders, agents, rates, watchlist, account, strategies


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init market data. Shutdown: cleanup."""
    from apps.classic.dashboard import MD
    app.state.md = MD(top_n=25)
    app.state.engine = None  # TUITradingEngine — initialized when trading starts
    app.state.ws_clients: list[WebSocket] = []
    yield
    # cleanup
    if app.state.engine:
        try:
            app.state.engine.stop()
        except Exception:
            pass


app = FastAPI(title="NEPSE TUI Bridge", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(lookup.router, prefix="/api/lookup", tags=["lookup"])
app.include_router(orders.router, prefix="/api/orders", tags=["orders"])
app.include_router(agents.router, prefix="/api/agent", tags=["agents"])
app.include_router(rates.router, prefix="/api/rates", tags=["rates"])
app.include_router(watchlist.router, prefix="/api/watchlist", tags=["watchlist"])
app.include_router(account.router, prefix="/api/account", tags=["account"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.websocket("/ws/quotes")
async def ws_quotes(websocket: WebSocket):
    """Push live quotes every 5 seconds."""
    await websocket.accept()
    app.state.ws_clients.append(websocket)
    try:
        while True:
            md = app.state.md
            try:
                md.refresh()
                quotes = {}
                if not md.quotes.empty:
                    for _, row in md.quotes.iterrows():
                        quotes[row["symbol"]] = float(row.get("ltp", 0))
                await websocket.send_json(quotes)
            except Exception:
                pass
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in app.state.ws_clients:
            app.state.ws_clients.remove(websocket)


@app.websocket("/ws/engine")
async def ws_engine(websocket: WebSocket):
    """Stream engine status and activity log."""
    await websocket.accept()
    try:
        while True:
            engine = app.state.engine
            if engine:
                try:
                    stats = engine.get_portfolio_stats()
                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "phase": stats.get("engine_phase", "idle"),
                            "nav": stats.get("nav", 0),
                            "positions": len(stats.get("positions", [])),
                            "regime": stats.get("regime", "unknown"),
                        },
                    })
                except Exception:
                    pass
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass
