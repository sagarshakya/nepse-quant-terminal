"""Canonical dataclasses returned by backend.core services.

Plain Python / numpy / pandas only. No UI imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np


@dataclass(slots=True)
class Quote:
    symbol: str
    last: float
    change: float
    change_pct: float
    volume: float
    high: float
    low: float
    open: float
    prev_close: float
    as_of: datetime
    sector: Optional[str] = None

    @property
    def is_up(self) -> bool:
        return self.change > 0

    @property
    def is_down(self) -> bool:
        return self.change < 0


@dataclass(slots=True)
class OHLCV:
    """Column-oriented OHLCV block for a single symbol. Fast to slice for charts."""
    symbol: str
    dates: np.ndarray        # dtype=datetime64[D]
    open: np.ndarray         # float64
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(self.dates.size)

    @property
    def empty(self) -> bool:
        return self.dates.size == 0


@dataclass(slots=True)
class MarketSnapshot:
    as_of: datetime
    quotes: list[Quote]
    index_change_pct: float = 0.0
    advancers: int = 0
    decliners: int = 0
    unchanged: int = 0


@dataclass(slots=True)
class IndexPoint:
    symbol: str
    last: float
    change_pct: float


@dataclass(slots=True)
class BacktestSummary:
    artifact_id: str
    name: str
    total_return: float
    sharpe: float
    max_dd: float
    n_trades: int
    path: str
    extras: dict = field(default_factory=dict)


@dataclass(slots=True)
class Signal:
    symbol: str
    score: float
    as_of: date
    regime: str = ""
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float
    last_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_pct: float
    sector: Optional[str] = None
