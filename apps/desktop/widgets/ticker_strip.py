"""Top scrolling marquee: indices, top movers, volume leaders, news headlines."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QRectF, QSize
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetricsF, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from apps.desktop.theme import (
    BG, PANE, BORDER, TEXT, TEXT_SECONDARY, TEXT_MUTED, GAIN, LOSS, ACCENT, WARN,
    mono_font, ui_font,
)
from backend.core.services import MarketService


# ---------------------------------------------------------------------------
# Segment model — each segment of the marquee is a list of (text, color) runs
# ---------------------------------------------------------------------------
Run = tuple[str, str]  # (text, hex color)


class _Marquee(QFrame):
    """Custom-painted horizontal scrolling marquee."""

    SEP = "  ◆  "
    SPEED_PX = 1.2  # pixels per tick (~60fps => ~72 px/s)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self._segments: list[list[Run]] = []  # list of segments; each segment = list of (text,color)
        self._cached_runs: list[Run] = []     # flattened (text,color) with separators
        self._total_w: float = 0.0
        self._offset: float = 0.0
        self._font = mono_font(12, 500)
        self._fm = QFontMetricsF(self._font)

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps
        self._timer.timeout.connect(self._advance)
        self._timer.start()

    def set_segments(self, segments: list[list[Run]]):
        self._segments = segments
        self._rebuild()

    def sizeHint(self) -> QSize:
        return QSize(400, 26)

    def _rebuild(self):
        flat: list[Run] = []
        for i, seg in enumerate(self._segments):
            if i > 0:
                flat.append((self.SEP, TEXT_MUTED))
            flat.extend(seg)
        self._cached_runs = flat
        self._total_w = sum(self._fm.horizontalAdvance(t) for t, _ in flat)
        # Start offset at right edge for first pass
        if self._offset == 0.0 and self._total_w > 0:
            self._offset = -float(self.width())

    def _advance(self):
        if self._total_w <= 0:
            return
        self._offset += self.SPEED_PX
        # wrap after scrolling the full string plus a gap
        if self._offset > self._total_w:
            self._offset = -float(self.width())
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.fillRect(self.rect(), QColor(PANE))
        if not self._cached_runs:
            return
        p.setFont(self._font)
        baseline_y = (self.height() + self._fm.ascent() - self._fm.descent()) / 2.0

        # draw twice (current pass and following pass) to simulate infinite scroll
        x = -self._offset
        self._draw_runs(p, x, baseline_y)
        # gap between cycles
        gap = self.width() * 0.5
        self._draw_runs(p, x + self._total_w + gap, baseline_y)

    def _draw_runs(self, p: QPainter, start_x: float, baseline_y: float):
        x = start_x
        if x + self._total_w < 0:
            return
        if x > self.width():
            return
        for text, color in self._cached_runs:
            w = self._fm.horizontalAdvance(text)
            if x + w > 0 and x < self.width():
                p.setPen(QPen(QColor(color)))
                p.drawText(QRectF(x, 0, w + 2, self.height()), Qt.AlignVCenter, text)
            x += w


class TickerStrip(QFrame):
    """28px always-on strip: scrolling marquee (movers + news) full-width,
    clock + session on the right."""

    def __init__(self, market: MarketService, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setStyleSheet(
            f"TickerStrip {{ background: {PANE}; border-bottom: 1px solid {BORDER}; }}"
        )
        self._market = market

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(0)

        # Scrolling marquee takes the full width
        self.marquee = _Marquee()
        lay.addWidget(self.marquee, 1)

        # Right side: clock + session
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet(f"color: {BORDER}; background: {BORDER};")
        sep2.setFixedWidth(1)
        sep2.setFixedHeight(16)
        lay.addSpacing(8)
        lay.addWidget(sep2)
        lay.addSpacing(8)

        self.clock = QLabel("—")
        self.clock.setFont(mono_font(11, 500))
        self.clock.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.session = QLabel("●LIVE")
        self.session.setFont(ui_font(11, 600))
        self.session.setStyleSheet(f"color: {GAIN}; letter-spacing: 1px;")
        lay.addWidget(self.clock)
        lay.addSpacing(10)
        lay.addWidget(self.session)

        # Timers
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_marquee)
        self._refresh_timer.start(120_000)
        QTimer.singleShot(200, self._refresh_marquee)

    # ---- rendering ----
    def _tick_clock(self):
        now = datetime.now()
        self.clock.setText(now.strftime("%a %H:%M:%S"))

    def _refresh_marquee(self):
        """Rebuild the scrolling text (indices + movers + volume leaders + news)."""
        segments: list[list[Run]] = []

        # Indices (NEPSE + sector indices) — lead the marquee
        try:
            index_points = self._market.index_strip()
        except Exception:
            index_points = []
        for ip in index_points:
            sign = "+" if ip.change_pct >= 0 else ""
            col = GAIN if ip.change_pct >= 0 else LOSS
            segments.append([
                (f"{ip.symbol} ", TEXT_SECONDARY),
                (f"{ip.last:,.2f} ", TEXT),
                (f"{sign}{ip.change_pct:.2f}%", col),
            ])

        # Movers
        try:
            gainers, losers, vol_top = self._market.top_movers(5)
        except Exception:
            gainers = losers = vol_top = []

        def arrow(q):
            return "▲" if q.change_pct >= 0 else "▼"

        for q in gainers:
            segments.append([
                ("GAIN ", WARN),
                (f"{arrow(q)} ", GAIN),
                (f"{q.symbol} ", TEXT),
                (f"{q.last:,.2f} ", TEXT_SECONDARY),
                (f"+{q.change_pct:.2f}%", GAIN),
            ])
        for q in losers:
            segments.append([
                ("LOSS ", WARN),
                (f"{arrow(q)} ", LOSS),
                (f"{q.symbol} ", TEXT),
                (f"{q.last:,.2f} ", TEXT_SECONDARY),
                (f"{q.change_pct:.2f}%", LOSS),
            ])
        for q in vol_top[:3]:
            segments.append([
                ("VOL ", WARN),
                (f"{q.symbol} ", TEXT),
                (f"{_fmt_vol(q.volume)}", ACCENT),
            ])

        # News headlines
        try:
            news = self._market.recent_news(limit=40)
        except Exception:
            news = []
        for n in news[:20]:
            label = n["sentiment"].lower()
            if label in ("positive", "bullish"):
                color = GAIN
            elif label in ("negative", "bearish"):
                color = LOSS
            else:
                color = TEXT_SECONDARY
            tag = f"NEWS"
            sym = n["symbol"] or n["source"] or ""
            head = n["headline"][:140]
            seg: list[Run] = [(f"{tag} ", WARN)]
            if sym:
                seg.append((f"{sym}: ", ACCENT))
            seg.append((head, color))
            segments.append(seg)

        # Fallback content if nothing is available
        if not segments:
            segments.append([
                ("WAITING ON DATA · scrapers haven't populated movers or news yet", TEXT_MUTED),
            ])

        self.marquee.set_segments(segments)


def _fmt_vol(v: float) -> str:
    if v >= 1e9: return f"{v/1e9:.2f}B"
    if v >= 1e6: return f"{v/1e6:.2f}M"
    if v >= 1e3: return f"{v/1e3:.1f}K"
    return f"{v:,.0f}"
