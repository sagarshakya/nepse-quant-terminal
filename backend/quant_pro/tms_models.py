"""
TMS execution model stubs.

Live brokerage execution (TMS19) is not included in this public release.
These stubs allow the codebase to import without errors.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class ExecutionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExecutionSource(str, Enum):
    SIGNAL = "signal"
    MANUAL = "manual"
    AGENT = "agent"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FILLED = "filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FillState(str, Enum):
    PARTIAL = "partial"
    FULL = "full"


@dataclass
class ExecutionIntent:
    intent_id: str = ""
    symbol: str = ""
    action: ExecutionAction = ExecutionAction.BUY
    source: ExecutionSource = ExecutionSource.SIGNAL
    status: ExecutionStatus = ExecutionStatus.PENDING
    shares: int = 0
    limit_price: Optional[float] = None
    created_at: str = ""
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    intent_id: str = ""
    success: bool = False
    filled_shares: int = 0
    fill_price: Optional[float] = None
    error: str = ""
    fill_state: FillState = FillState.FULL


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
