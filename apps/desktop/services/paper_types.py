"""Paper-trading dataclasses used by PaperService and all desktop workspaces."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class PaperPosition:
    symbol: str
    qty: float
    avg_price: float
    last_price: float
    day_pnl: float
    day_pct: float
    unrealized_pnl: float
    pct_return: float
    days_held: int
    signal_type: str
    sector: str
    market_value: float
    cost_basis: float
    weight_pct: float


@dataclass(slots=True)
class NavSummary:
    nav: float
    cash: float
    invested: float
    day_pnl: float
    day_pct: float
    total_return: float
    gross_return: float
    nepse_return: float
    alpha: float
    max_dd: float
    n_positions: int


@dataclass(slots=True)
class PaperOrder:
    id: str
    symbol: str
    action: str          # "BUY" | "SELL"
    qty: int
    order_price: float
    fill_price: Optional[float]
    status: str          # "OPEN" | "FILLED" | "CANCELLED"
    created_at: str
    updated_at: str
    reason: str
    slippage_pct: float = 2.0


@dataclass(slots=True)
class Trade:
    date: str
    action: str
    symbol: str
    shares: float
    price: float
    fees: float
    pnl: float
    pnl_pct: float
    reason: str


@dataclass(slots=True)
class ConcentrationRow:
    row_type: str        # "POSITION" | "SECTOR"
    name: str
    value: float
    weight_pct: float


@dataclass(slots=True)
class PaperSignal:
    symbol: str
    score: float
    signal_type: str
    strength: float
    confidence: float
    regime: str
    as_of: str


@dataclass(slots=True)
class StrategyEntry:
    id: str
    name: str
    source: str          # "builtin" | "custom"
    description: str
    notes: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    is_active: bool = False
