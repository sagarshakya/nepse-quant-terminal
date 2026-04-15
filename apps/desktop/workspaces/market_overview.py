"""Market Overview workspace.

Layout:
    ┌──────────────────────────────┬────────────────────────────┐
    │  Chart pane (candles + vol)  │  Sector breadth list       │
    ├──────────────────────────────┴────────────────────────────┤
    │  Market grid (virtualized, 380+ rows)                     │
    └───────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget,
)

from apps.desktop.theme import PANE, BORDER, TEXT_SECONDARY, TEXT, GAIN, LOSS, ui_font, mono_font
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.market_grid import MarketGrid
from apps.desktop.widgets.chart_pane import ChartPane
from backend.core.services import MarketService


class SectorBreadthPane(QFrame):
    def __init__(self, market: MarketService, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._market = market
        self.setStyleSheet(f"SectorBreadthPane {{ background: {PANE}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        header = QLabel("SECTORS")
        header.setFont(ui_font(11, 600))
        header.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 1px;")
        lay.addWidget(header)

        self._rows_container = QVBoxLayout()
        self._rows_container.setSpacing(4)
        lay.addLayout(self._rows_container)
        lay.addStretch(1)
        self.refresh()

    def refresh(self):
        # clear previous rows
        while self._rows_container.count():
            item = self._rows_container.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        data = self._market.sector_breadth()  # from benchmark_index_history via index_strip-like
        # Prefer the richer index_strip output
        for ip in self._market.index_strip():
            row = QFrame()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(8)
            name = QLabel(ip.symbol)
            name.setFont(ui_font(12))
            name.setStyleSheet(f"color: {TEXT};")
            value = QLabel(f"{ip.last:,.2f}")
            value.setFont(mono_font(12))
            value.setStyleSheet(f"color: {TEXT_SECONDARY};")
            pct = QLabel(f"{'+' if ip.change_pct >= 0 else ''}{ip.change_pct:.2f}%")
            pct.setFont(mono_font(12, 500))
            pct.setStyleSheet(f"color: {GAIN if ip.change_pct >= 0 else LOSS};")
            hl.addWidget(name)
            hl.addStretch(1)
            hl.addWidget(value)
            hl.addWidget(pct)
            self._rows_container.addWidget(row)


class MarketOverview(QWidget):
    """Composite widget for the Market Overview workspace."""

    def __init__(self, market: MarketService, link: LinkGroup, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._market = market
        self._link = link

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        v_split = QSplitter(Qt.Vertical)
        v_split.setHandleWidth(1)

        # Top: chart + sector side
        top = QSplitter(Qt.Horizontal)
        top.setHandleWidth(1)
        self.chart = ChartPane(market, link)
        self.sectors = SectorBreadthPane(market)
        top.addWidget(self.chart)
        top.addWidget(self.sectors)
        top.setStretchFactor(0, 4)
        top.setStretchFactor(1, 1)
        top.setSizes([900, 260])

        # Bottom: the grid
        self.grid = MarketGrid(market, link)

        v_split.addWidget(top)
        v_split.addWidget(self.grid)
        v_split.setStretchFactor(0, 3)
        v_split.setStretchFactor(1, 2)
        v_split.setSizes([440, 360])

        lay.addWidget(v_split)
