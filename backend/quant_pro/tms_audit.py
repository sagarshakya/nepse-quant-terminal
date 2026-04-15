"""
TMS audit/persistence stubs.

Live brokerage execution (TMS19) is not included in this public release.
These stubs allow the codebase to import without errors.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_live_audit_db_path() -> Path:
    return Path("data/runtime/live_audit.db")


def init_live_audit_db() -> None:
    pass


def load_execution_intent(intent_id: str) -> Optional[Dict[str, Any]]:
    return None


def save_execution_intent(intent: Any) -> None:
    pass


def update_execution_intent(intent_id: str, **kwargs: Any) -> None:
    pass


def count_intents_for_day(date_str: str) -> int:
    return 0


def find_recent_open_intent(symbol: str, action: str) -> Optional[Dict[str, Any]]:
    return None


def mark_intent_notified(intent_id: str) -> None:
    pass


def load_latest_live_orders() -> List[Dict[str, Any]]:
    return []


def load_latest_live_positions() -> List[Dict[str, Any]]:
    return []


def load_latest_tms_snapshot(snapshot_name: str) -> Optional[Dict[str, Any]]:
    return None


def save_tms_snapshot(snapshot_name: str, payload: Dict[str, Any]) -> None:
    pass
