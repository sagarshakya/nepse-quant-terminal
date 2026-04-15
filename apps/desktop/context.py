"""Link-group context. Central ticker/date state; panes subscribe via Qt signals."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from PySide6.QtCore import QObject, Signal


@dataclass(slots=True)
class Context:
    symbol: Optional[str] = None
    start: Optional[date] = None
    end: Optional[date] = None


class LinkGroup(QObject):
    """A link group is a shared Context that panes subscribe to. Changes fan
    out to every connected pane in one Qt event-loop tick."""
    changed = Signal(object)   # emits Context
    symbol_changed = Signal(str)

    def __init__(self, name: str = "A", color: str = "#4D9FFF"):
        super().__init__()
        self.name = name
        self.color = color
        self._ctx = Context()

    @property
    def context(self) -> Context:
        return self._ctx

    def set_symbol(self, symbol: str):
        if not symbol:
            return
        symbol = symbol.strip().upper()
        if symbol == self._ctx.symbol:
            return
        self._ctx.symbol = symbol
        self.symbol_changed.emit(symbol)
        self.changed.emit(self._ctx)

    def set_range(self, start: Optional[date], end: Optional[date]):
        self._ctx.start = start
        self._ctx.end = end
        self.changed.emit(self._ctx)
