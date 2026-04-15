"""Professional candlestick + volume chart with toggle-able MAs and an
indicator subplot selector (MACD / Bollinger / Stochastic / OBV / ATR / VWAP).

Layout:
    ┌───────────────────────────────────────────────────────────────┐
    │ PaneHeader                                                    │
    ├───────────────────────────────────────────────────────────────┤
    │ meta: SYMBOL LAST CHG   [HI LO BARS]   1W 1M 3M 6M 1Y 3Y ALL  │
    ├───────────────────────────────────────────────────────────────┤
    │ toolbar: D/W  MA20 MA50 MA200 BB VWAP   ·  Indicator: [MACD▼] │
    ├───────────────────────────────────────────────────────────────┤
    │ price plot (candles + optional overlays)                      │
    ├───────────────────────────────────────────────────────────────┤
    │ volume plot                                                   │
    ├───────────────────────────────────────────────────────────────┤
    │ indicator plot (optional — shown only when selected)          │
    └───────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QAction, QPainter, QPen, QBrush, QColor, QPicture
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QToolButton,
    QVBoxLayout, QWidget,
)

from apps.desktop.theme import (
    BG, PANE, ELEVATED, BORDER, BORDER_STRONG, TEXT, TEXT_SECONDARY, TEXT_MUTED,
    GAIN, LOSS, WARN, ACCENT, mono_font, ui_font,
)
from apps.desktop.utils import fmt_number, fmt_signed, fmt_signed_pct
from apps.desktop.context import LinkGroup
from apps.desktop.widgets.pane_header import PaneHeader
from backend.core.services import MarketService
from backend.core.types import OHLCV


pg.setConfigOption("background", BG)
pg.setConfigOption("foreground", TEXT_SECONDARY)
pg.setConfigOption("antialias", True)


INDICATOR_CHOICES = [
    ("none",       "— none —"),
    ("macd",       "MACD (12/26/9)"),
    ("rsi",        "RSI (14)"),
    ("stoch",      "Stochastic (14,3)"),
    ("obv",        "OBV"),
    ("atr",        "ATR (14)"),
    ("volosc",     "Volume Osc."),
]


# ---------------------------------------------------------------------------
# Indicator math
# ---------------------------------------------------------------------------
def _sma(a: np.ndarray, window: int) -> np.ndarray:
    if a.size < window:
        return np.full(a.size, np.nan, dtype=np.float64)
    c = np.cumsum(np.insert(a.astype(np.float64), 0, 0.0))
    ma = (c[window:] - c[:-window]) / window
    pad = np.full(window - 1, np.nan, dtype=np.float64)
    return np.concatenate([pad, ma])


def _ema(a: np.ndarray, span: int) -> np.ndarray:
    if a.size == 0:
        return a.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(a, dtype=np.float64)
    out[0] = a[0]
    for i in range(1, a.size):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def _macd(close: np.ndarray):
    if close.size < 26:
        nan = np.full(close.size, np.nan)
        return nan, nan, nan
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.size
    if n < period + 1:
        return np.full(n, np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.zeros(n)
    avg_l = np.zeros(n)
    avg_g[period] = gain[1:period + 1].mean()
    avg_l[period] = loss[1:period + 1].mean()
    for i in range(period + 1, n):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period
    rs = avg_g / np.where(avg_l == 0, 1.0, avg_l)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi


def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray, k: int = 14, d: int = 3):
    n = close.size
    if n < k:
        return np.full(n, np.nan), np.full(n, np.nan)
    lo_min = np.full(n, np.nan)
    hi_max = np.full(n, np.nan)
    for i in range(k - 1, n):
        lo_min[i] = low[i - k + 1:i + 1].min()
        hi_max[i] = high[i - k + 1:i + 1].max()
    pctk = 100.0 * (close - lo_min) / np.where(hi_max - lo_min == 0, 1.0, hi_max - lo_min)
    pctd = _sma(pctk, d)
    return pctk, pctd


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    if close.size == 0:
        return close.copy()
    delta = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(delta * volume)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.size
    if n < 2:
        return np.full(n, np.nan)
    prev = np.roll(close, 1); prev[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _bollinger(close: np.ndarray, period: int = 20, k: float = 2.0):
    ma = _sma(close, period)
    n = close.size
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        std[i] = close[i - period + 1:i + 1].std(ddof=0)
    return ma, ma + k * std, ma - k * std


def _vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling-window VWAP (daily bars — true intraday VWAP needs tick data)."""
    tp = (high + low + close) / 3.0
    pv = tp * volume
    n = close.size
    out = np.full(n, np.nan)
    if n < window:
        return out
    csum_pv = np.cumsum(pv)
    csum_v  = np.cumsum(volume)
    for i in range(window - 1, n):
        num = csum_pv[i] - (csum_pv[i - window] if i >= window else 0.0)
        den = csum_v[i]  - (csum_v[i - window]  if i >= window else 0.0)
        out[i] = num / den if den else np.nan
    return out


def _resample_weekly(ohlcv: OHLCV) -> OHLCV:
    """Resample daily OHLCV to weekly bars, grouped by ISO-week."""
    if ohlcv.empty:
        return ohlcv
    dates = ohlcv.dates.astype("datetime64[D]")
    # Group key = Monday-of-week
    dow = (dates.astype("int64") - np.datetime64("1970-01-05").astype("int64")) % 7
    week_start = dates - dow.astype("timedelta64[D]")
    # Boundaries where week_start changes
    changes = np.concatenate([[True], week_start[1:] != week_start[:-1]])
    starts = np.flatnonzero(changes)
    ends = np.append(starts[1:], dates.size)

    w_dates = week_start[starts]
    w_open   = ohlcv.open[starts]
    w_close  = ohlcv.close[ends - 1]
    w_high   = np.array([ohlcv.high[s:e].max()  for s, e in zip(starts, ends)], dtype=np.float64)
    w_low    = np.array([ohlcv.low[s:e].min()   for s, e in zip(starts, ends)], dtype=np.float64)
    w_volume = np.array([ohlcv.volume[s:e].sum() for s, e in zip(starts, ends)], dtype=np.float64)
    return OHLCV(
        symbol=ohlcv.symbol, dates=w_dates,
        open=w_open, high=w_high, low=w_low, close=w_close, volume=w_volume,
    )


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------
class _CandleItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self._picture: Optional[QPicture] = None
        self._bounds = QRectF(0, 0, 0, 0)

    def set_data(self, xs, o, h, l, c):
        if xs.size == 0:
            self._picture = None
            self._bounds = QRectF(0, 0, 0, 0)
            self.update()
            return
        pic = QPicture()
        p = QPainter(pic)
        p.setRenderHint(QPainter.Antialiasing, False)
        w = 0.72
        gain_pen = QPen(QColor(GAIN)); gain_pen.setWidthF(1.0); gain_pen.setCosmetic(True)
        loss_pen = QPen(QColor(LOSS)); loss_pen.setWidthF(1.0); loss_pen.setCosmetic(True)
        gain_brush = QBrush(QColor(GAIN))
        loss_brush = QBrush(QColor(LOSS))
        for i in range(xs.size):
            x = xs[i]
            oi, hi, li, ci = o[i], h[i], l[i], c[i]
            if ci >= oi:
                p.setPen(gain_pen); p.setBrush(gain_brush)
            else:
                p.setPen(loss_pen); p.setBrush(loss_brush)
            p.drawLine(QPointF(x, li), QPointF(x, hi))
            top, bot = max(oi, ci), min(oi, ci)
            body = max(top - bot, 1e-9)
            p.drawRect(QRectF(x - w / 2, bot, w, body))
        p.end()
        self._picture = pic
        xmin = float(xs.min()) - 1
        xmax = float(xs.max()) + 1
        ymin = float(np.nanmin(l))
        ymax = float(np.nanmax(h))
        pad = (ymax - ymin) * 0.05 or 1.0
        self._bounds = QRectF(xmin, ymin - pad, xmax - xmin, (ymax - ymin) + 2 * pad)
        self.informViewBoundsChanged()
        self.update()

    def paint(self, p: QPainter, *args):
        if self._picture is not None:
            p.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        return self._bounds


class _VolumeItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self._picture: Optional[QPicture] = None
        self._bounds = QRectF(0, 0, 0, 0)

    def set_data(self, xs, vol, up_mask):
        if xs.size == 0:
            self._picture = None
            self._bounds = QRectF(0, 0, 0, 0)
            self.update()
            return
        pic = QPicture()
        p = QPainter(pic)
        p.setPen(QPen(Qt.NoPen))
        w = 0.72
        up_c = QColor(GAIN); up_c.setAlpha(140)
        dn_c = QColor(LOSS); dn_c.setAlpha(140)
        up_b = QBrush(up_c); dn_b = QBrush(dn_c)
        for i in range(xs.size):
            p.setBrush(up_b if up_mask[i] else dn_b)
            p.drawRect(QRectF(xs[i] - w / 2, 0.0, w, float(vol[i])))
        p.end()
        self._picture = pic
        self._bounds = QRectF(float(xs.min()) - 1, 0.0, float(xs.max() - xs.min()) + 2, float(vol.max()) * 1.05)
        self.informViewBoundsChanged()
        self.update()

    def paint(self, p: QPainter, *args):
        if self._picture is not None:
            p.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        return self._bounds


class _DateAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dates: np.ndarray = np.array([], dtype="datetime64[D]")
        self._mode: str = "date"  # "date" or "intraday"

    def set_dates(self, dates: np.ndarray, mode: str = "date"):
        self._dates = dates
        self._mode = mode

    def tickStrings(self, values, scale, spacing):
        out = []
        n = self._dates.size
        if self._mode == "intraday":
            prev_hour = None
            for v in values:
                i = int(round(v))
                if 0 <= i < n:
                    t = self._dates[i].astype("datetime64[m]").astype("O")
                    if prev_hour != t.hour or spacing >= 60:
                        out.append(t.strftime("%H:%M"))
                    else:
                        out.append(t.strftime(":%M"))
                    prev_hour = t.hour
                else:
                    out.append("")
            return out

        prev_year = None
        for v in values:
            i = int(round(v))
            if 0 <= i < n:
                d = self._dates[i].astype("datetime64[D]").astype("O")
                if prev_year != d.year and spacing >= 20:
                    out.append(d.strftime("%Y"))
                else:
                    out.append(d.strftime("%b %d"))
                prev_year = d.year
            else:
                out.append("")
        return out


# ---------------------------------------------------------------------------
# Toggle button
# ---------------------------------------------------------------------------
def _toggle_btn(label: str, tooltip: str = "", color: str = ACCENT) -> QPushButton:
    b = QPushButton(label)
    b.setCheckable(True)
    b.setFixedHeight(20)
    b.setFont(mono_font(10, 500))
    b.setToolTip(tooltip or label)
    b.setStyleSheet(f"""
        QPushButton {{ background: transparent; border: 1px solid {BORDER};
                       color: {TEXT_SECONDARY}; padding: 0 8px; }}
        QPushButton:hover {{ color: {TEXT}; border-color: {TEXT_MUTED}; }}
        QPushButton:checked {{ color: {color}; border-color: {color}; }}
    """)
    return b


# ---------------------------------------------------------------------------
# ChartPane
# ---------------------------------------------------------------------------
class ChartPane(QFrame):
    """Candlestick chart with volume + toggle-able indicators."""

    def __init__(
        self,
        market: MarketService,
        link: LinkGroup,
        *,
        show_header: bool = True,
        show_volume: bool = True,
        default_window_bars: int = 132,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._market = market
        self._link = link
        self._symbol: Optional[str] = None
        self._default_window = default_window_bars
        self._resolution = "D"   # "I", "D", or "W"
        self._ohlcv_raw: Optional[OHLCV] = None
        self._intraday_raw: Optional[OHLCV] = None
        self.setStyleSheet(f"ChartPane {{ background: {BG}; }}")

        # auto-refresh timer for intraday — fires only when resolution == "I"
        self._intraday_timer = QTimer(self)
        self._intraday_timer.setInterval(20_000)
        self._intraday_timer.timeout.connect(self._refresh_intraday)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Header row --------------------------------------------------
        if show_header:
            self.header = PaneHeader("Price", link_color=link.color)
            self._sub = QLabel("")
            self._sub.setFont(mono_font(12, 500))
            self._sub.setStyleSheet(f"color: {TEXT};")
            self.header.add_action(self._sub)
            root.addWidget(self.header)

            # Big symbol row under header
            meta = QFrame()
            meta.setStyleSheet(f"QFrame {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}")
            meta.setFixedHeight(52)
            ml = QHBoxLayout(meta)
            ml.setContentsMargins(14, 6, 14, 6)
            ml.setSpacing(14)

            self.sym_label = QLabel("—")
            self.sym_label.setFont(ui_font(20, 600))
            self.sym_label.setStyleSheet(f"color: {TEXT};")

            self.last_label = QLabel("")
            self.last_label.setFont(mono_font(20, 500))
            self.last_label.setStyleSheet(f"color: {TEXT};")

            self.change_label = QLabel("")
            self.change_label.setFont(mono_font(13, 500))

            self.range_label = QLabel("")
            self.range_label.setFont(mono_font(11))
            self.range_label.setStyleSheet(f"color: {TEXT_SECONDARY};")

            ml.addWidget(self.sym_label)
            ml.addWidget(self.last_label)
            ml.addWidget(self.change_label)
            ml.addStretch(1)
            ml.addWidget(self.range_label)
            ml.addSpacing(10)

            # Range selector buttons on the right
            self._range_btns: list[QPushButton] = []
            for label, bars in (("1W", 5), ("1M", 22), ("3M", 66), ("6M", 132),
                                ("1Y", 252), ("3Y", 252 * 3), ("ALL", 0)):
                b = QPushButton(label)
                b.setCheckable(True)
                b.setFixedHeight(22)
                b.setFixedWidth(36)
                b.setFont(mono_font(10, 500))
                b.setStyleSheet(f"""
                    QPushButton {{ background: transparent; border: 1px solid {BORDER}; color: {TEXT_SECONDARY}; padding: 0; }}
                    QPushButton:hover {{ color: {TEXT}; border-color: {TEXT_MUTED}; }}
                    QPushButton:checked {{ color: {ACCENT}; border-color: {ACCENT}; }}
                """)
                b.clicked.connect(lambda _c=False, bb=bars, w=b: self._set_window(bb, w))
                self._range_btns.append(b)
                ml.addWidget(b)

            root.addWidget(meta)

            # --- Toolbar row with resolution + overlay toggles + indicator selector ---
            tool = QFrame()
            tool.setStyleSheet(f"QFrame {{ background: {PANE}; border-bottom: 1px solid {BORDER}; }}")
            tool.setFixedHeight(30)
            tl = QHBoxLayout(tool)
            tl.setContentsMargins(14, 4, 14, 4)
            tl.setSpacing(6)

            # Resolution I / D / W
            self._res_btns: dict[str, QPushButton] = {}
            for code, tip in (
                ("I", "Intraday 5-min bars (live)"),
                ("D", "Daily bars"),
                ("W", "Weekly bars (resampled)"),
            ):
                b = _toggle_btn(code, tooltip=tip, color=WARN)
                b.setFixedWidth(28)
                b.clicked.connect(lambda _c=False, cc=code, bb=b: self._set_resolution(cc, bb))
                self._res_btns[code] = b
                tl.addWidget(b)
            self._res_btns["D"].setChecked(True)

            tl.addSpacing(12)
            sep = QFrame(); sep.setFrameShape(QFrame.VLine)
            sep.setStyleSheet(f"background: {BORDER};"); sep.setFixedHeight(16)
            tl.addWidget(sep)
            tl.addSpacing(8)

            # Hidden backing buttons — kept so existing overlay logic (which
            # checks btn.isChecked()) can remain untouched. Swapped style for
            # menu-based UI below.
            self.btn_ma20  = _toggle_btn("MA20",  "", "#8AB8E2"); self.btn_ma20.hide()
            self.btn_ma50  = _toggle_btn("MA50",  "", "#C9A94A"); self.btn_ma50.hide()
            self.btn_ma200 = _toggle_btn("MA200", "", "#B087D6"); self.btn_ma200.hide()
            self.btn_bb    = _toggle_btn("BB",    "", ACCENT);    self.btn_bb.hide()
            self.btn_vwap  = _toggle_btn("VWAP",  "", WARN);      self.btn_vwap.hide()

            # Dropdown menu exposing overlays
            overlay_btn = QToolButton()
            overlay_btn.setText("Overlays  ▾")
            overlay_btn.setFont(ui_font(11, 500))
            overlay_btn.setCursor(Qt.PointingHandCursor)
            overlay_btn.setFixedHeight(22)
            overlay_btn.setStyleSheet(f"""
                QToolButton {{
                    background: transparent; color: {TEXT_SECONDARY};
                    border: 1px solid {BORDER}; padding: 0 10px;
                }}
                QToolButton:hover {{ color: {TEXT}; border-color: {TEXT_MUTED}; }}
                QToolButton::menu-indicator {{ image: none; width: 0; }}
            """)
            overlay_btn.setPopupMode(QToolButton.InstantPopup)

            menu = QMenu(overlay_btn)
            menu.setStyleSheet(f"""
                QMenu {{ background: {PANE}; color: {TEXT};
                         border: 1px solid {BORDER}; padding: 4px 0; }}
                QMenu::item {{ padding: 6px 18px 6px 26px; }}
                QMenu::item:selected {{ background: {ELEVATED}; }}
                QMenu::indicator {{ left: 8px; width: 12px; height: 12px; }}
                QMenu::separator {{ height: 1px; background: {BORDER};
                                    margin: 4px 10px; }}
            """)
            self._overlay_actions: list[tuple[QAction, QPushButton]] = []
            for label, btn in (
                ("MA 20",           self.btn_ma20),
                ("MA 50",           self.btn_ma50),
                ("MA 200",          self.btn_ma200),
                ("Bollinger Bands", self.btn_bb),
                ("VWAP",            self.btn_vwap),
            ):
                act = QAction(label, menu)
                act.setCheckable(True)
                act.setChecked(False)
                def _on_toggled(checked, _btn=btn):
                    _btn.setChecked(checked)
                    self._refresh_overlays()
                act.toggled.connect(_on_toggled)
                menu.addAction(act)
                self._overlay_actions.append((act, btn))

            menu.addSeparator()
            self._vol_action = QAction("Volume", menu)
            self._vol_action.setCheckable(True)
            self._vol_action.setChecked(True)
            self._vol_action.toggled.connect(self._set_volume_visible)
            menu.addAction(self._vol_action)

            overlay_btn.setMenu(menu)
            tl.addWidget(overlay_btn)

            tl.addStretch(1)

            # Indicator selector
            lbl = QLabel("Indicator")
            lbl.setFont(ui_font(10, 600))
            lbl.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1px;")
            tl.addWidget(lbl)
            self.ind_combo = QComboBox()
            self.ind_combo.setFont(mono_font(11))
            self.ind_combo.setFixedHeight(22)
            self.ind_combo.setMinimumWidth(150)
            for key, label in INDICATOR_CHOICES:
                self.ind_combo.addItem(label, key)
            self.ind_combo.setStyleSheet(f"""
                QComboBox {{ background: transparent; color: {TEXT};
                             border: 1px solid {BORDER}; padding: 0 8px; }}
                QComboBox:hover {{ border-color: {TEXT_MUTED}; }}
                QComboBox::drop-down {{ border: none; width: 18px; }}
                QComboBox QAbstractItemView {{ background: {PANE}; color: {TEXT};
                                               border: 1px solid {BORDER}; }}
            """)
            self.ind_combo.currentIndexChanged.connect(self._refresh_indicator)
            tl.addWidget(self.ind_combo)

            root.addWidget(tool)
        else:
            self.header = None
            self.sym_label = self.last_label = self.change_label = self.range_label = None
            self._range_btns = []
            self._res_btns = {}
            self.btn_ma20 = self.btn_ma50 = self.btn_ma200 = self.btn_bb = self.btn_vwap = None
            self.ind_combo = None

        # --- Plot area ---------------------------------------------------
        self._layout_widget = pg.GraphicsLayoutWidget()
        self._layout_widget.setBackground(BG)
        self._layout_widget.ci.setContentsMargins(0, 0, 0, 0)
        self._layout_widget.ci.setSpacing(0)
        root.addWidget(self._layout_widget, 1)

        self._date_axis = _DateAxis(orientation="bottom")
        self._price_axis_right = pg.AxisItem(orientation="right")
        self._price_plot: pg.PlotItem = self._layout_widget.addPlot(
            row=0, col=0,
            axisItems={
                "bottom": pg.AxisItem("bottom"),
                "right":  self._price_axis_right,
            },
        )
        self._style_plot(self._price_plot)
        self._price_plot.getAxis("bottom").setStyle(showValues=False)
        self._price_plot.getAxis("bottom").setHeight(0)
        self._price_plot.hideAxis("left")
        self._price_plot.showAxis("right")
        self._price_plot.getAxis("right").setWidth(64)

        self._candles = _CandleItem()
        self._price_plot.addItem(self._candles)

        # Overlays — created but hidden by default
        self._ma20 = pg.PlotCurveItem(pen=pg.mkPen(QColor("#8AB8E2"), width=1.2))
        self._ma50 = pg.PlotCurveItem(pen=pg.mkPen(QColor("#C9A94A"), width=1.2))
        self._ma200 = pg.PlotCurveItem(pen=pg.mkPen(QColor("#B087D6"), width=1.2))
        self._bb_mid = pg.PlotCurveItem(pen=pg.mkPen(QColor(ACCENT), width=1.0, style=Qt.DashLine))
        self._bb_up  = pg.PlotCurveItem(pen=pg.mkPen(QColor(ACCENT), width=1.0))
        self._bb_lo  = pg.PlotCurveItem(pen=pg.mkPen(QColor(ACCENT), width=1.0))
        self._vwap_curve = pg.PlotCurveItem(pen=pg.mkPen(QColor(WARN), width=1.2))
        for item in (self._ma20, self._ma50, self._ma200,
                     self._bb_mid, self._bb_up, self._bb_lo, self._vwap_curve):
            self._price_plot.addItem(item)
            item.setVisible(False)

        # Volume plot
        if show_volume:
            self._layout_widget.nextRow()
            self._vol_plot: pg.PlotItem = self._layout_widget.addPlot(
                row=1, col=0,
                axisItems={
                    "bottom": self._date_axis,
                    "right":  pg.AxisItem("right"),
                },
            )
            self._style_plot(self._vol_plot)
            self._vol_plot.setMaximumHeight(110)
            self._vol_plot.setMinimumHeight(70)
            self._vol_plot.hideAxis("left")
            self._vol_plot.showAxis("right")
            self._vol_plot.getAxis("right").setWidth(64)
            self._vol_plot.setMouseEnabled(x=True, y=False)
            self._vol_plot.hideButtons()
            self._vol_plot.setXLink(self._price_plot)
            self._volume = _VolumeItem()
            self._vol_plot.addItem(self._volume)
        else:
            self._vol_plot = None
            self._volume = None

        # Indicator subplot (created once, lazily populated)
        self._layout_widget.nextRow()
        self._ind_plot: pg.PlotItem = self._layout_widget.addPlot(
            row=2, col=0,
            axisItems={"right": pg.AxisItem("right"), "bottom": pg.AxisItem("bottom")},
        )
        self._style_plot(self._ind_plot)
        self._ind_plot.hideAxis("left")
        self._ind_plot.showAxis("right")
        self._ind_plot.getAxis("right").setWidth(64)
        self._ind_plot.getAxis("bottom").setStyle(showValues=False)
        self._ind_plot.getAxis("bottom").setHeight(0)
        self._ind_plot.setMaximumHeight(140)
        self._ind_plot.setMinimumHeight(90)
        self._ind_plot.setMouseEnabled(x=True, y=True)
        self._ind_plot.hideButtons()
        self._ind_plot.setXLink(self._price_plot)
        self._ind_plot.setVisible(False)
        self._ind_curves: list = []     # pyqtgraph items owned by the indicator plot
        self._ind_lines: list = []      # horizontal reference lines
        self._ind_title = pg.TextItem("", anchor=(0, 0), color=QColor(TEXT_SECONDARY))
        self._ind_title.setFont(mono_font(10, 500))
        self._ind_plot.addItem(self._ind_title, ignoreBounds=True)

        # Crosshair
        pen_cross = pg.mkPen(QColor(TEXT_MUTED), style=Qt.DashLine, width=1)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen_cross)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen_cross)
        self._vline.setZValue(1000); self._hline.setZValue(1000)
        self._price_plot.addItem(self._vline, ignoreBounds=True)
        self._price_plot.addItem(self._hline, ignoreBounds=True)

        self._price_label = pg.TextItem(anchor=(0, 0.5), color=QColor(TEXT))
        self._price_label.setFont(mono_font(11, 500))
        self._price_plot.addItem(self._price_label, ignoreBounds=True)

        self._proxy = pg.SignalProxy(
            self._price_plot.scene().sigMouseMoved,
            rateLimit=60, slot=self._on_mouse_moved,
        )

        self._link.symbol_changed.connect(self.set_symbol)

        if show_header and self._range_btns:
            # default: 6M
            self._range_btns[3].setChecked(True)

    # ---- styling ----
    def _style_plot(self, plot: pg.PlotItem):
        plot.showGrid(x=True, y=True, alpha=0.12)
        for side in ("bottom", "left", "right", "top"):
            ax = plot.getAxis(side)
            ax.setPen(QColor(BORDER))
            ax.setTextPen(QColor(TEXT_SECONDARY))
            ax.setStyle(tickFont=mono_font(10), tickTextOffset=6)
        plot.setMouseEnabled(x=True, y=True)
        plot.hideButtons()

    # ---- public ----
    def set_symbol(self, symbol: str):
        if not symbol:
            return
        self._symbol = symbol
        self._ohlcv_raw = self._market.history(symbol)
        if self._resolution == "I":
            self._intraday_raw = self._market.intraday(symbol)
        self._render()

    def _set_window(self, bars: int, button: QPushButton):
        for b in self._range_btns:
            b.setChecked(b is button)
        self._default_window = bars if bars > 0 else 0
        self._render()

    def _set_resolution(self, code: str, button: QPushButton):
        for c, b in self._res_btns.items():
            b.setChecked(b is button)
        self._resolution = code
        if code == "I":
            if self._symbol:
                self._intraday_raw = self._market.intraday(self._symbol)
            self._intraday_timer.start()
        else:
            self._intraday_timer.stop()
        self._render()

    def _refresh_intraday(self):
        if self._resolution != "I" or not self._symbol:
            return
        self._intraday_raw = self._market.intraday(self._symbol)
        self._render()

    def _set_volume_visible(self, visible: bool):
        if self._vol_plot is None:
            return
        self._vol_plot.setVisible(visible)
        if visible:
            self._vol_plot.setMaximumHeight(110)
            self._vol_plot.setMinimumHeight(70)
        else:
            self._vol_plot.setMaximumHeight(0)
            self._vol_plot.setMinimumHeight(0)

    # ---- rendering ----
    def _active_ohlcv(self) -> OHLCV:
        if self._resolution == "I":
            raw = self._intraday_raw
            if raw is None:
                return OHLCV(
                    symbol=self._symbol or "",
                    dates=np.array([], dtype="datetime64[m]"),
                    open=np.array([]), high=np.array([]), low=np.array([]),
                    close=np.array([]), volume=np.array([]),
                )
            return raw
        raw = self._ohlcv_raw
        if raw is None or raw.empty:
            return raw if raw is not None else OHLCV(
                symbol=self._symbol or "",
                dates=np.array([], dtype="datetime64[D]"),
                open=np.array([]), high=np.array([]), low=np.array([]),
                close=np.array([]), volume=np.array([]),
            )
        if self._resolution == "W":
            return _resample_weekly(raw)
        return raw

    def _render(self):
        ohlcv = self._active_ohlcv()
        if ohlcv.empty:
            if self.sym_label:
                self.sym_label.setText(self._symbol or "—")
                self.last_label.setText("—")
                self.change_label.setText("no data")
                self.change_label.setStyleSheet(f"color: {WARN};")
                self.range_label.setText("")
            self._candles.set_data(*(np.array([], np.float64) for _ in range(5)))
            if self._volume is not None:
                self._volume.set_data(np.array([], np.float64), np.array([], np.float64), np.array([], bool))
            for c in (self._ma20, self._ma50, self._ma200, self._bb_mid, self._bb_up, self._bb_lo, self._vwap_curve):
                c.setData([], [])
            self._ind_plot.setVisible(False)
            return

        # keep recent N bars per selected window; always keep extra history for MA200/Boll
        n = len(ohlcv)
        if self._resolution == "I":
            start = 0  # show full intraday session
        else:
            keep_extra = 200 if self._resolution == "D" else 50
            if self._default_window and n > self._default_window + keep_extra:
                start = max(0, n - (self._default_window + keep_extra))
            else:
                start = 0
        dates = ohlcv.dates[start:]
        o = ohlcv.open[start:]; h = ohlcv.high[start:]; l = ohlcv.low[start:]
        c = ohlcv.close[start:]; v = ohlcv.volume[start:]
        xs = np.arange(len(dates), dtype=np.float64)
        self._date_axis.set_dates(dates, mode="intraday" if self._resolution == "I" else "date")

        self._candles.set_data(xs, o, h, l, c)
        if self._volume is not None:
            self._volume.set_data(xs, v, c >= o)

        self._xs = xs
        self._hlc_slice = (h, l, c, v)
        self._refresh_overlays()
        self._refresh_indicator()

        # view range: intraday shows entire session; otherwise selected window
        if self._resolution == "I":
            left, right = 0, len(xs) - 1
        elif self._default_window:
            right = len(xs) - 1
            left = max(0, right - self._default_window + 1)
        else:
            left, right = 0, len(xs) - 1
        # Pad short windows so candles don't render as giant blocks.
        # Aim for >= MIN_SLOTS visible slots; extra space becomes whitespace.
        vis = max(1, right - left + 1)
        MIN_SLOTS = 30
        if vis < MIN_SLOTS:
            pad_frac = (MIN_SLOTS - vis) / (2.0 * vis)
        else:
            pad_frac = 0.02
        self._price_plot.setXRange(left, right, padding=pad_frac)
        # autoscale y to visible slice
        y_slice_lo = float(np.nanmin(l[left:right + 1]))
        y_slice_hi = float(np.nanmax(h[left:right + 1]))
        pad = (y_slice_hi - y_slice_lo) * 0.05 or 1.0
        self._price_plot.setYRange(y_slice_lo - pad, y_slice_hi + pad, padding=0)
        if self._vol_plot is not None:
            vol_max = float(np.nanmax(v[left:right + 1])) * 1.15 or 1.0
            self._vol_plot.setYRange(0, vol_max, padding=0)

        # Header labels
        last = float(c[-1])
        prev = float(c[-2]) if c.size > 1 else last
        chg = last - prev
        chg_pct = (chg / prev * 100.0) if prev else 0.0
        hi = float(np.nanmax(h))
        lo = float(np.nanmin(l))
        if self.sym_label:
            self.sym_label.setText(self._symbol or "")
            self.last_label.setText(fmt_number(last))
            self.change_label.setText(f"{fmt_signed(chg)}   {fmt_signed_pct(chg_pct)}")
            self.change_label.setStyleSheet(f"color: {GAIN if chg >= 0 else LOSS};")
            res_tag = self._resolution
            self.range_label.setText(f"HI {fmt_number(hi)}   LO {fmt_number(lo)}   BARS {c.size}{res_tag}")
        if self.header is not None:
            self.header.set_subtitle(self._symbol or "")

    # ---- overlays ----
    def _refresh_overlays(self):
        if not hasattr(self, "_xs"):
            return
        xs = self._xs
        h, l, c, v = self._hlc_slice

        def show(curve, on: bool, arr: np.ndarray):
            curve.setVisible(on)
            if on:
                curve.setData(xs, arr)
            else:
                curve.setData([], [])

        if self.btn_ma20 is not None:
            show(self._ma20,  self.btn_ma20.isChecked()  and c.size >= 20,  _sma(c, 20)  if c.size >= 20 else np.array([]))
            show(self._ma50,  self.btn_ma50.isChecked()  and c.size >= 50,  _sma(c, 50)  if c.size >= 50 else np.array([]))
            show(self._ma200, self.btn_ma200.isChecked() and c.size >= 200, _sma(c, 200) if c.size >= 200 else np.array([]))

            bb_on = self.btn_bb.isChecked() and c.size >= 20
            if bb_on:
                mid, up, lo = _bollinger(c, 20, 2.0)
                show(self._bb_mid, True, mid); show(self._bb_up, True, up); show(self._bb_lo, True, lo)
            else:
                for curve in (self._bb_mid, self._bb_up, self._bb_lo):
                    show(curve, False, np.array([]))

            vw_on = self.btn_vwap.isChecked() and c.size >= 20
            show(self._vwap_curve, vw_on, _vwap(h, l, c, v, 20) if vw_on else np.array([]))

    # ---- indicator subplot ----
    def _clear_indicator(self):
        for it in self._ind_curves:
            self._ind_plot.removeItem(it)
        for it in self._ind_lines:
            self._ind_plot.removeItem(it)
        self._ind_curves = []
        self._ind_lines = []

    def _add_hline(self, y: float, color: str, dash: bool = True):
        pen = pg.mkPen(QColor(color), width=1,
                       style=Qt.DashLine if dash else Qt.SolidLine)
        ln = pg.InfiniteLine(pos=y, angle=0, pen=pen)
        self._ind_plot.addItem(ln, ignoreBounds=True)
        self._ind_lines.append(ln)

    def _refresh_indicator(self):
        if self.ind_combo is None or not hasattr(self, "_xs"):
            return
        key = self.ind_combo.currentData()
        self._clear_indicator()
        if key == "none" or key is None:
            self._ind_plot.setVisible(False)
            return

        xs = self._xs
        h, l, c, v = self._hlc_slice
        self._ind_plot.setVisible(True)
        self._ind_title.setPos(xs[0] if xs.size else 0, 0)

        if key == "macd":
            macd, signal, hist = _macd(c)
            # histogram as bar-like line
            bar = pg.BarGraphItem(x=xs, height=hist, width=0.72, brush=QColor(ACCENT))
            self._ind_plot.addItem(bar)
            self._ind_curves.append(bar)
            m_curve = pg.PlotCurveItem(xs, macd, pen=pg.mkPen(QColor("#8AB8E2"), width=1.3))
            s_curve = pg.PlotCurveItem(xs, signal, pen=pg.mkPen(QColor(WARN), width=1.2, style=Qt.DashLine))
            for c2 in (m_curve, s_curve):
                self._ind_plot.addItem(c2); self._ind_curves.append(c2)
            self._add_hline(0.0, TEXT_MUTED, dash=False)
            self._ind_title.setPlainText("MACD (12,26,9)")
            self._ind_plot.enableAutoRange(axis="y", enable=True)

        elif key == "rsi":
            r = _rsi(c, 14)
            curve = pg.PlotCurveItem(xs, r, pen=pg.mkPen(QColor(ACCENT), width=1.3))
            self._ind_plot.addItem(curve); self._ind_curves.append(curve)
            self._add_hline(70, LOSS); self._add_hline(30, GAIN); self._add_hline(50, TEXT_MUTED)
            self._ind_title.setPlainText("RSI (14)")
            self._ind_plot.setYRange(0, 100, padding=0)

        elif key == "stoch":
            k_line, d_line = _stoch(h, l, c, 14, 3)
            k_curve = pg.PlotCurveItem(xs, k_line, pen=pg.mkPen(QColor(ACCENT), width=1.3))
            d_curve = pg.PlotCurveItem(xs, d_line, pen=pg.mkPen(QColor(WARN), width=1.2, style=Qt.DashLine))
            for c2 in (k_curve, d_curve):
                self._ind_plot.addItem(c2); self._ind_curves.append(c2)
            self._add_hline(80, LOSS); self._add_hline(20, GAIN)
            self._ind_title.setPlainText("Stochastic %K/%D (14,3)")
            self._ind_plot.setYRange(0, 100, padding=0)

        elif key == "obv":
            o = _obv(c, v)
            curve = pg.PlotCurveItem(xs, o, pen=pg.mkPen(QColor(ACCENT), width=1.3))
            self._ind_plot.addItem(curve); self._ind_curves.append(curve)
            self._ind_title.setPlainText("On-Balance Volume")
            self._ind_plot.enableAutoRange(axis="y", enable=True)

        elif key == "atr":
            a = _atr(h, l, c, 14)
            curve = pg.PlotCurveItem(xs, a, pen=pg.mkPen(QColor(WARN), width=1.3))
            self._ind_plot.addItem(curve); self._ind_curves.append(curve)
            self._ind_title.setPlainText("ATR (14)")
            self._ind_plot.enableAutoRange(axis="y", enable=True)

        elif key == "volosc":
            short = _sma(v, 10); long_ = _sma(v, 30)
            osc = (short - long_) / np.where(long_ == 0, 1.0, long_) * 100.0
            curve = pg.PlotCurveItem(xs, osc, pen=pg.mkPen(QColor(ACCENT), width=1.3))
            self._ind_plot.addItem(curve); self._ind_curves.append(curve)
            self._add_hline(0, TEXT_MUTED, dash=False)
            self._ind_title.setPlainText("Volume Oscillator (10/30)")
            self._ind_plot.enableAutoRange(axis="y", enable=True)

    # ---- mouse readout ----
    def _on_mouse_moved(self, event):
        pos = event[0]
        if not self._price_plot.sceneBoundingRect().contains(pos):
            return
        view = self._price_plot.vb
        mp = view.mapSceneToView(pos)
        x = mp.x(); y = mp.y()
        self._vline.setPos(x)
        self._hline.setPos(y)
        dates = self._date_axis._dates
        i = int(round(x))
        if 0 <= i < len(dates):
            d = dates[i].astype("datetime64[D]").astype("O")
            if self.range_label:
                self.range_label.setText(f"{d:%Y-%m-%d}   y={y:,.2f}")
        self._price_label.setText(f" {fmt_number(y)} ")
        self._price_label.setPos(view.viewRange()[0][1], y)
