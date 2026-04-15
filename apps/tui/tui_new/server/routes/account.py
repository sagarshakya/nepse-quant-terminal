"""Account management endpoints — paper account registry."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.quant_pro.paths import ensure_dir, get_trading_runtime_dir

router = APIRouter()

PAPER_ACCOUNTS_DIR = ensure_dir(Path(get_trading_runtime_dir(__file__)) / "accounts")
PAPER_ACCOUNTS_REGISTRY = PAPER_ACCOUNTS_DIR / "registry.json"


def _load_registry() -> dict:
    if PAPER_ACCOUNTS_REGISTRY.exists():
        try:
            data = json.loads(PAPER_ACCOUNTS_REGISTRY.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"accounts": [], "active_id": None}


def _save_registry(reg: dict):
    PAPER_ACCOUNTS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    PAPER_ACCOUNTS_REGISTRY.write_text(json.dumps(reg, indent=2, default=str))


@router.get("")
async def get_accounts(request: Request):
    reg = _load_registry()
    active_id = reg.get("active_id")
    return [
        {
            "id": a.get("id", ""),
            "name": a.get("name", ""),
            "nav": float(a.get("nav", 1_000_000)),
            "cash": float(a.get("cash", 1_000_000)),
            "is_active": a.get("id") == active_id,
            "created_at": a.get("created_at", ""),
        }
        for a in reg.get("accounts", [])
    ]


class AccountAction(BaseModel):
    action: str  # "create", "activate"
    name: str = ""
    capital: float = 1_000_000
    id: str = ""


@router.post("")
async def manage_account(body: AccountAction, request: Request):
    reg = _load_registry()

    if body.action == "create":
        new_id = uuid.uuid4().hex[:8]
        new_account = {
            "id": new_id,
            "name": body.name or f"Account-{new_id}",
            "nav": body.capital,
            "cash": body.capital,
            "created_at": datetime.now().isoformat(),
        }
        reg.setdefault("accounts", []).append(new_account)
        if not reg.get("active_id"):
            reg["active_id"] = new_id
        _save_registry(reg)
        return {
            "id": new_id,
            "name": new_account["name"],
            "nav": body.capital,
            "cash": body.capital,
            "is_active": reg["active_id"] == new_id,
            "created_at": new_account["created_at"],
        }

    elif body.action == "activate":
        ids = {a["id"] for a in reg.get("accounts", [])}
        if body.id in ids:
            reg["active_id"] = body.id
            _save_registry(reg)
            return {"status": "activated", "active_id": body.id}
        return {"status": "not_found"}

    return {"status": "unknown_action"}
