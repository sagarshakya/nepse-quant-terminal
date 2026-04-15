"""Agent analysis and chat endpoints — wraps agent_analyst."""
from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


@router.get("/analysis")
async def get_analysis(request: Request):
    try:
        from backend.agents.agent_analyst import load_agent_analysis, build_algo_shortlist_snapshot

        analysis = load_agent_analysis()
        if not analysis:
            return {"picks": [], "regime": "unknown", "timestamp": ""}

        picks = []
        for sym, data in analysis.items():
            if not isinstance(data, dict):
                continue
            picks.append({
                "symbol": sym,
                "decision": str(data.get("decision", "HOLD")),
                "score": float(data.get("score", 0)),
                "confidence": float(data.get("confidence", 0)),
                "reasoning": str(data.get("reasoning", "")),
                "red_flags": data.get("red_flags", []),
                "catalysts": data.get("catalysts", []),
            })

        # Sort by score descending
        picks.sort(key=lambda x: x["score"], reverse=True)

        return {
            "picks": picks[:10],
            "regime": "unknown",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        return {"picks": [], "regime": "unknown", "timestamp": ""}


@router.post("/analyze")
async def trigger_analysis(request: Request):
    try:
        from backend.agents.agent_analyst import analyze as agent_analyze

        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: agent_analyze(force=True))
        return {"status": "analysis_complete"}
    except Exception as e:
        return {"status": f"error: {str(e)}"}


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def agent_chat(body: ChatRequest, request: Request):
    try:
        from backend.agents.agent_analyst import (
            load_agent_history,
            append_external_agent_chat_message,
        )

        # For now return a simple response
        # Full agent chat requires subprocess spawning
        history = load_agent_history() or []
        recent = history[-5:] if len(history) > 5 else history

        return {
            "reply": f"Agent chat received: '{body.message}'. Full agent integration coming soon.",
            "history_count": len(history),
        }
    except Exception as e:
        return {"reply": f"Agent unavailable: {str(e)}"}
