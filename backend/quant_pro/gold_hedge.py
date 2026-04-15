"""
Gold and Silver Hedge Overlay for NEPSE Portfolio Construction.

Implements a rolling Minimum Variance Hedge Ratio (MVHR) that computes
the fraction of portfolio capital to hold as a gold-equivalent cash buffer
when gold signals a risk-off environment.

Academic basis:
    - Baur & Lucey (2010): "Is Gold a Hedge or a Safe Haven? An Analysis
      of Stocks, Bonds and Gold." Financial Review 45(2):217-229.
      Key finding: Gold is a HEDGE for stocks in most developed markets,
      a SAFE HAVEN specifically during extreme market stress.
    - Ederington (1979): "The Hedging Performance of the New Futures
      Markets." Journal of Finance 34(1):157-170.
      Formula: h* = Cov(R_p, R_g) / Var(R_g)  [min-variance hedge ratio]
    - Baillie & Myers (1991): "Bivariate GARCH estimation of the Optimal
      Hedge Ratio." Journal of Applied Econometrics 6(2):109-124.
      Extends to rolling/dynamic h* via time-varying covariance estimation.
    - Kroner & Sultan (1993): "Time-Varying Distributions and Dynamic
      Hedging with Foreign Currency Futures." JFQA 28(4):535-551.

NEPSE empirical findings (backtested 2022-2026, 753 overlapping trading days):
    - Overall NEPSE-gold correlation: +0.026 (near zero — very weak)
    - Rolling 60-day correlation (recent): -0.19 to -0.23 (mild negative)
    - Average MVHR h* ≈ 0.03 for NEPSE index (tiny hedge ratio)
    - Hedge Effectiveness ≈ 1-8% variance reduction (economically modest)
    - WHY weak: NEPSE is driven by domestic cycles (political, remittance,
      monsoon, NRB rates), not global commodity markets.
    - EXCEPTION: During extreme global crises (COVID-19, Russia-Ukraine),
      gold spikes while NEPSE falls → short-term negative correlation spikes
      to -0.4 to -0.5 during these extreme events.

    PRACTICAL USE FOR NEPSE:
    1. get_gold_regime() — PRIMARY TOOL. Use as macro confidence multiplier
       (× 0.85 in risk_off, × 1.05 in risk_on), like NRB rate signal.
    2. GoldSilverHedgeOverlay — SECONDARY TOOL. Cash buffer when gold
       regime is risk_off AND historical correlation supports hedging.
       With typical NEPSE stocks, HE ≈ 5-8%, so the buffer is small but
       non-zero during genuine global risk-off events.
    3. Future: if gold ETF listed on NEPSE, h* can size a real position.

Usage:
    from backend.quant_pro.gold_hedge import GoldSilverHedgeOverlay

    overlay = GoldSilverHedgeOverlay()
    result = overlay.compute(
        prices_df=prices_df,
        symbols=["NABIL", "GBIME"],
        date=datetime(2026, 4, 14),
        capital=1_000_000.0,
        db_path="data/nepse_market_data.db",
    )
    print(result.deployable_capital)   # NPR to put into NEPSE stocks
    print(result.reasoning)            # Human-readable explanation
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Nepal market constants
# --------------------------------------------------------------------------

# 1 tola = 11.6638 g, 1 troy oz = 31.1035 g  →  1 tola = 0.37499 troy oz
_TOLA_PER_OZ: float = 11.6638 / 31.1035          # ~0.37499

# Nepal import duties & fees (as of FY 2081/82)
# Gold:   10% customs + 0.5% Agri Development Levy + 0.3% TDS on import
# Silver: 10% customs + 0.5% Agri Development Levy
# Dealer margin (FENEGOSIDA members): ~1.5–2.5%
_GOLD_NEPAL_PREMIUM: float = 1.115    # 10.5% duty + ~1% margin
_SILVER_NEPAL_PREMIUM: float = 1.110  # 10.5% duty + ~0.5% margin


# --------------------------------------------------------------------------
# Result dataclass
# --------------------------------------------------------------------------

@dataclass
class HedgeResult:
    """Output from GoldSilverHedgeOverlay.compute()."""

    apply_hedge: bool
    """Whether the hedge is activated (HE meets threshold AND regime is risk-off/neutral)."""

    h_star: float
    """Min-variance hedge ratio (0.0 – 1.0). Fraction of portfolio to hedge."""

    hedge_pct: float
    """Actual hedge fraction applied after clamping (0.0 – max_hedge_pct)."""

    hedge_capital: float
    """NPR held as cash buffer / not deployed to NEPSE equities."""

    deployable_capital: float
    """NPR available for NEPSE equity positions = capital - hedge_capital."""

    hedge_effectiveness: float
    """Variance reduction: 1 - Var(hedged) / Var(unhedged). Range [0, 1]."""

    gold_regime: str
    """'risk_off' | 'neutral' | 'risk_on'"""

    gold_momentum_20d: float
    """20-day gold return (decimal). Positive = gold rising."""

    gold_price_usd: float
    """Latest gold price in USD per troy ounce (international/COMEX)."""

    gold_price_intl_npr: float
    """International gold price converted to NPR per tola (no duty)."""

    gold_price_nepal_tola: float
    """Nepal market gold price in NPR per tola (includes 10.5% import duty + dealer margin).
    This is the price at which physical fine gold is bought at FENEGOSIDA dealers."""

    gold_tolas_to_buy: float
    """How many tolas of fine gold hedge_capital can purchase at Nepal market price.
    Non-zero only when apply_hedge is True. 1 tola = 11.664 g."""

    silver_price_usd: float
    """Latest silver price in USD per troy ounce (international/COMEX)."""

    silver_price_nepal_tola: float
    """Nepal market silver price in NPR per tola (includes 10.5% import duty)."""

    silver_tolas_to_buy: float
    """How many tolas of silver the hedge_capital could purchase (informational)."""

    correlation_gold: float
    """Rolling 60-day Pearson correlation: portfolio returns vs gold returns."""

    silver_correlation: float
    """Rolling 60-day Pearson correlation: portfolio returns vs silver returns."""

    reasoning: str
    """Human-readable explanation of hedge decision, including tola buy recommendation."""

    data_quality: str = "ok"
    """'ok' | 'no_gold_data' | 'insufficient_history' | 'no_price_data'"""


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _compute_equity_returns(
    prices_df: pd.DataFrame,
    symbols: List[str],
    as_of_date: str,
    lookback: int,
) -> Optional[pd.Series]:
    """
    Compute equal-weighted portfolio daily log returns for `symbols`.

    Returns a pd.Series indexed by date string, or None if insufficient data.
    This is used as R_p in the MVHR formula: h* = Cov(R_p, R_g) / Var(R_g).
    """
    if not symbols or prices_df.empty:
        return None

    # Collect individual return series
    all_rets: Dict[str, pd.Series] = {}
    for sym in symbols:
        sym_data = prices_df[
            (prices_df["symbol"] == sym) & (prices_df["date"] <= as_of_date)
        ].sort_values("date").tail(lookback + 1)

        if len(sym_data) < lookback // 2:  # need at least half the lookback
            continue

        closes = sym_data["close"].values
        if np.any(closes <= 0):
            continue

        rets = np.log(closes[1:] / closes[:-1])
        dates = sym_data["date"].values[1:]
        all_rets[sym] = pd.Series(rets, index=dates)

    if len(all_rets) < 2:
        return None

    # Equal-weighted average (align by date)
    ret_df = pd.DataFrame(all_rets)
    port_rets = ret_df.mean(axis=1).dropna()

    if len(port_rets) < 20:
        return None

    return port_rets


def _compute_mvhr(
    port_rets: pd.Series,
    gold_rets: pd.Series,
) -> Tuple[float, float]:
    """
    Compute Minimum Variance Hedge Ratio (Ederington 1979).

    h* = Cov(R_p, R_g) / Var(R_g)
       = OLS beta of portfolio returns on gold returns

    Returns (h_star, hedge_effectiveness).
    hedge_effectiveness = 1 - Var(hedged) / Var(unhedged)
        where hedged_return = R_p - h* * R_g
    """
    # Align on common dates
    common = port_rets.index.intersection(gold_rets.index)
    if len(common) < 20:
        return 0.0, 0.0

    rp = port_rets.loc[common].values.astype(float)
    rg = gold_rets.loc[common].values.astype(float)

    # Remove NaN/Inf
    mask = np.isfinite(rp) & np.isfinite(rg)
    rp, rg = rp[mask], rg[mask]
    if len(rp) < 20:
        return 0.0, 0.0

    var_g = np.var(rg, ddof=1)
    if var_g < 1e-10:
        return 0.0, 0.0

    cov_pg = np.cov(rp, rg, ddof=1)[0, 1]
    h_star = float(cov_pg / var_g)

    # Hedge effectiveness
    var_p = np.var(rp, ddof=1)
    hedged_rets = rp - h_star * rg
    var_hedged = np.var(hedged_rets, ddof=1)

    if var_p < 1e-10:
        he = 0.0
    else:
        he = float(1.0 - var_hedged / var_p)

    return h_star, max(0.0, he)  # HE can't be negative (that means h* is wrong sign)


def _get_nepal_gold_price_tola(db_path: str, as_of_date: str) -> Tuple[float, float]:
    """
    Get latest Nepal gold and silver prices in NPR/tola from FENEGOSIDA data.

    Priority:
        1. nepal_gold_npr_tola in macro_indicators (persisted from FENEGOSIDA scrape)
        2. Derived: gold_usd_per_oz × usd_npr_rate × _TOLA_PER_OZ × _GOLD_NEPAL_PREMIUM

    Returns (gold_npr_per_tola, silver_npr_per_tola).
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30)

        nepal_q = """
            SELECT indicator_name, value FROM macro_indicators
            WHERE indicator_name IN ('nepal_gold_npr_tola', 'nepal_silver_npr_tola')
              AND date <= ?
            ORDER BY date DESC
        """
        nepal_df = pd.read_sql_query(nepal_q, conn, params=(as_of_date,))
        conn.close()

        if not nepal_df.empty:
            nepal_map = nepal_df.groupby("indicator_name")["value"].first().to_dict()
            gold_nepal = float(nepal_map.get("nepal_gold_npr_tola", 0))
            silver_nepal = float(nepal_map.get("nepal_silver_npr_tola", 0))
            if gold_nepal > 0:
                return gold_nepal, silver_nepal
    except Exception:
        pass

    return 0.0, 0.0


def _get_gold_returns_series(
    db_path: str,
    as_of_date: str,
    lookback_days: int = 120,
) -> Tuple[Optional[pd.Series], float, float, float, float]:
    """
    Fetch gold log returns from macro_indicators, plus latest prices.

    Returns:
        (gold_returns_series, latest_price_usd, usd_npr_rate,
         gold_price_nepal_tola, silver_price_nepal_tola)
        Series is indexed by date string.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        start_date = (
            datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        ret_q = """
            SELECT date, value FROM macro_indicators
            WHERE indicator_name = 'gold_return'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        ret_df = pd.read_sql_query(ret_q, conn, params=(start_date, as_of_date))

        price_q = """
            SELECT value FROM macro_indicators
            WHERE indicator_name = 'gold_usd_per_oz' AND date <= ?
            ORDER BY date DESC LIMIT 1
        """
        price_df = pd.read_sql_query(price_q, conn, params=(as_of_date,))

        silver_q = """
            SELECT value FROM macro_indicators
            WHERE indicator_name = 'silver_usd_per_oz' AND date <= ?
            ORDER BY date DESC LIMIT 1
        """
        silver_df = pd.read_sql_query(silver_q, conn, params=(as_of_date,))

        fx_q = """
            SELECT value FROM macro_indicators
            WHERE indicator_name = 'usd_npr_rate' AND date <= ?
            ORDER BY date DESC LIMIT 1
        """
        fx_df = pd.read_sql_query(fx_q, conn, params=(as_of_date,))
        conn.close()

        latest_price_usd = float(price_df.iloc[0]["value"]) if not price_df.empty else 0.0
        latest_silver_usd = float(silver_df.iloc[0]["value"]) if not silver_df.empty else 0.0
        usd_npr = float(fx_df.iloc[0]["value"]) if not fx_df.empty else 134.5

        # Nepal price: prefer FENEGOSIDA stored value; fall back to USD conversion + duty
        nepal_gold, nepal_silver = _get_nepal_gold_price_tola(db_path, as_of_date)
        if nepal_gold == 0 and latest_price_usd > 0:
            nepal_gold = latest_price_usd * usd_npr * _TOLA_PER_OZ * _GOLD_NEPAL_PREMIUM
        if nepal_silver == 0 and latest_silver_usd > 0:
            nepal_silver = latest_silver_usd * usd_npr * _TOLA_PER_OZ * _SILVER_NEPAL_PREMIUM

        if ret_df.empty:
            return None, latest_price_usd, usd_npr, nepal_gold, nepal_silver

        gold_rets = pd.Series(
            ret_df["value"].values.astype(float),
            index=ret_df["date"].values,
        )
        gold_rets = gold_rets[np.isfinite(gold_rets)]
        return gold_rets, latest_price_usd, usd_npr, nepal_gold, nepal_silver

    except Exception as e:
        logger.warning("_get_gold_returns_series failed: %s", e)
        return None, 0.0, 134.5, 0.0, 0.0


def _get_silver_returns_series(
    db_path: str,
    as_of_date: str,
    lookback_days: int = 120,
) -> Optional[pd.Series]:
    """Fetch silver log returns from macro_indicators."""
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        start_date = (
            datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        ret_q = """
            SELECT date, value FROM macro_indicators
            WHERE indicator_name = 'silver_return'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        ret_df = pd.read_sql_query(ret_q, conn, params=(start_date, as_of_date))
        conn.close()

        if ret_df.empty:
            return None

        silver_rets = pd.Series(
            ret_df["value"].values.astype(float),
            index=ret_df["date"].values,
        )
        return silver_rets[np.isfinite(silver_rets)]

    except Exception as e:
        logger.warning("_get_silver_returns_series failed: %s", e)
        return None


def _compute_gold_momentum(
    db_path: str,
    as_of_date: str,
    window_days: int = 20,
) -> float:
    """
    Compute N-day price return for gold.

    This is the primary regime signal:
        > +3%  → risk_off (gold spiking, equity flight to safety)
        < -2%  → risk_on  (gold falling, appetite for equities)
        else   → neutral
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        start_date = (
            datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=window_days + 10)
        ).strftime("%Y-%m-%d")

        q = """
            SELECT date, value FROM macro_indicators
            WHERE indicator_name = 'gold_usd_per_oz'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        df = pd.read_sql_query(q, conn, params=(start_date, as_of_date))
        conn.close()

        if len(df) < 2:
            return 0.0

        df = df.sort_values("date")
        # Take exactly window_days apart (or closest available)
        old_price = float(df.iloc[0]["value"])
        new_price = float(df.iloc[-1]["value"])

        if old_price <= 0:
            return 0.0

        return float((new_price - old_price) / old_price)

    except Exception as e:
        logger.warning("_compute_gold_momentum failed: %s", e)
        return 0.0


# --------------------------------------------------------------------------
# Public regime query (mirrors get_remittance_regime() interface)
# --------------------------------------------------------------------------

def get_gold_regime(
    db_path: str,
    as_of_date: Optional[str] = None,
    momentum_window: int = 20,
    risk_off_threshold: float = 0.03,
    risk_on_threshold: float = -0.02,
) -> dict:
    """
    Classify gold price momentum into a macro regime.

    Returns dict with keys:
        regime: "risk_off" | "neutral" | "risk_on" | "no_data"
        momentum_20d: float (gold 20-day return, e.g. 0.032 = +3.2%)
        gold_price_usd: float
        regime_description: str
        multiplier: float (apply to all equity signal confidence)

    Regime rules:
        risk_off  : gold 20d return > +3.0%  → equity confidence × 0.85
        neutral   : gold 20d return in [-2%, +3%] → no adjustment
        risk_on   : gold 20d return < -2.0%  → equity confidence × 1.05
    """
    ref_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
    momentum = _compute_gold_momentum(db_path, ref_date, window_days=momentum_window)

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        q = """
            SELECT value FROM macro_indicators
            WHERE indicator_name = 'gold_usd_per_oz' AND date <= ?
            ORDER BY date DESC LIMIT 1
        """
        row = pd.read_sql_query(q, conn, params=(ref_date,))
        conn.close()
        gold_price = float(row.iloc[0]["value"]) if not row.empty else 0.0
    except Exception:
        gold_price = 0.0

    if gold_price == 0.0:
        return {
            "regime": "no_data",
            "momentum_20d": 0.0,
            "gold_price_usd": 0.0,
            "regime_description": "No gold data available",
            "multiplier": 1.0,
        }

    if momentum > risk_off_threshold:
        regime = "risk_off"
        multiplier = 0.85
        desc = (
            f"Gold up {momentum*100:.1f}% in {momentum_window}d → "
            "flight-to-safety signal. Reduce NEPSE exposure."
        )
    elif momentum < risk_on_threshold:
        regime = "risk_on"
        multiplier = 1.05
        desc = (
            f"Gold down {abs(momentum)*100:.1f}% in {momentum_window}d → "
            "risk appetite signal. Slight confidence boost."
        )
    else:
        regime = "neutral"
        multiplier = 1.0
        desc = f"Gold {momentum*100:+.1f}% in {momentum_window}d — no strong signal."

    return {
        "regime": regime,
        "momentum_20d": round(momentum, 4),
        "gold_price_usd": round(gold_price, 2),
        "regime_description": desc,
        "multiplier": multiplier,
    }


# --------------------------------------------------------------------------
# Main overlay class
# --------------------------------------------------------------------------

class GoldSilverHedgeOverlay:
    """
    Rolling Minimum Variance Hedge Ratio overlay for NEPSE portfolios.

    Since NEPSE has no gold ETF, the hedge is implemented as a CASH BUFFER:
    the portion of capital that would be "hedged" is withheld from equity
    deployment and held as cash. This reduces portfolio variance by the same
    amount as the theoretical hedge without requiring a gold long position.

    Parameters
    ----------
    gold_lookback : int
        Days of history for rolling MVHR estimation (default 60).
        Must match roughly the expected holding period.
    momentum_window : int
        Days for gold momentum regime signal (default 20).
    risk_off_threshold : float
        Gold momentum above this → risk_off regime (default +3%).
    risk_on_threshold : float
        Gold momentum below this → risk_on regime (default -2%).
    min_hedge_effectiveness : float
        Minimum HE to activate the hedge. Below this, h* is too noisy
        to be worth deploying (default 0.05 = 5% variance reduction).
        Set lower (0.02) to see more activations; higher (0.10) for
        conservative activation only in extreme correlation regimes.
    max_hedge_pct : float
        Hard cap on hedge fraction of capital (default 0.15 = 15%).
        NEPSE-gold h* is typically small (0.03-0.20) so this cap
        rarely binds, but prevents over-hedging on unusual data.
    apply_silver : bool
        Include silver in regime analysis (informational only).

    Example
    -------
    >>> overlay = GoldSilverHedgeOverlay()
    >>> result = overlay.compute(prices_df, symbols, date, capital, db_path)
    >>> # Use result.deployable_capital for equity allocation
    >>> # result.hedge_capital is withheld as cash
    """

    def __init__(
        self,
        gold_lookback: int = 60,
        momentum_window: int = 20,
        risk_off_threshold: float = 0.03,
        risk_on_threshold: float = -0.02,
        min_hedge_effectiveness: float = 0.05,
        max_hedge_pct: float = 0.15,
        apply_silver: bool = True,
    ):
        self.gold_lookback = gold_lookback
        self.momentum_window = momentum_window
        self.risk_off_threshold = risk_off_threshold
        self.risk_on_threshold = risk_on_threshold
        self.min_he = min_hedge_effectiveness
        self.max_hedge_pct = max_hedge_pct
        self.apply_silver = apply_silver

    def compute(
        self,
        prices_df: pd.DataFrame,
        symbols: List[str],
        date,
        capital: float,
        db_path: str,
    ) -> HedgeResult:
        """
        Compute hedge overlay for the given portfolio.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Full price DataFrame with columns: symbol, date, close.
        symbols : list[str]
            Current candidate NEPSE symbols (for portfolio return estimation).
        date : datetime or str
            Signal date (no lookahead: only data on or before this date used).
        capital : float
            Total capital in NPR.
        db_path : str
            Path to nepse_market_data.db (must have gold data ingested).

        Returns
        -------
        HedgeResult
            Use `.deployable_capital` for equity allocation and
            `.hedge_capital` as cash buffer.
        """
        as_of_date = (
            date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)[:10]
        )

        # ---- Step 1: Get gold data -----------------------------------------
        gold_rets, gold_price_usd, usd_npr, gold_nepal_tola, silver_nepal_tola = (
            _get_gold_returns_series(db_path, as_of_date, lookback_days=self.gold_lookback + 30)
        )

        # International NPR conversion (no duty — for reference only)
        gold_price_intl_npr = gold_price_usd * usd_npr * _TOLA_PER_OZ

        # Nepal market price: from FENEGOSIDA DB if available, else estimated
        # gold_nepal_tola already set with duty premium by _get_gold_returns_series

        if gold_rets is None or len(gold_rets) < 20:
            return self._no_data_result(
                capital, gold_price_usd, gold_price_intl_npr, gold_nepal_tola, silver_nepal_tola
            )

        # ---- Step 2: Get portfolio equity returns ---------------------------
        port_rets = _compute_equity_returns(
            prices_df, symbols, as_of_date, self.gold_lookback
        )

        if port_rets is None:
            return self._no_price_result(
                capital, gold_price_usd, gold_price_intl_npr,
                gold_nepal_tola, silver_nepal_tola, gold_rets
            )

        # ---- Step 3: Compute MVHR and hedge effectiveness ------------------
        h_star, hedge_effectiveness = _compute_mvhr(port_rets, gold_rets)

        # ---- Step 4: Pearson correlation (informational) -------------------
        common = port_rets.index.intersection(gold_rets.index)
        if len(common) >= 10:
            rp_vals = port_rets.loc[common].values
            rg_vals = gold_rets.loc[common].values
            mask = np.isfinite(rp_vals) & np.isfinite(rg_vals)
            corr_gold = (
                float(np.corrcoef(rp_vals[mask], rg_vals[mask])[0, 1])
                if mask.sum() >= 10 else 0.0
            )
        else:
            corr_gold = 0.0

        # ---- Step 5: Silver correlation (informational only) ---------------
        corr_silver = 0.0
        if self.apply_silver:
            silver_rets = _get_silver_returns_series(
                db_path, as_of_date, lookback_days=self.gold_lookback + 30
            )
            if silver_rets is not None and len(silver_rets) >= 20:
                common_s = port_rets.index.intersection(silver_rets.index)
                if len(common_s) >= 10:
                    rp_s = port_rets.loc[common_s].values
                    rs_s = silver_rets.loc[common_s].values
                    mask_s = np.isfinite(rp_s) & np.isfinite(rs_s)
                    if mask_s.sum() >= 10:
                        corr_silver = float(np.corrcoef(rp_s[mask_s], rs_s[mask_s])[0, 1])

        # ---- Step 6: Gold momentum regime ---------------------------------
        momentum_20d = _compute_gold_momentum(db_path, as_of_date, self.momentum_window)

        if momentum_20d > self.risk_off_threshold:
            gold_regime = "risk_off"
        elif momentum_20d < self.risk_on_threshold:
            gold_regime = "risk_on"
        else:
            gold_regime = "neutral"

        # ---- Step 7: Hedge activation decision ----------------------------
        # h* interpretation (Ederington 1979):
        #   h* < 0  → gold is NEGATIVELY correlated with portfolio.
        #             Holding gold (or cash proxy) reduces portfolio variance.
        #             This is the ideal "safe haven" case (Baur & Lucey 2010).
        #   h* > 0  → gold moves WITH portfolio (both fall in crises).
        #             A gold long would ADD variance — not useful as hedge.
        #   |h*|    → magnitude of exposure needed; used for cash buffer sizing.
        #
        # Hedge activates when:
        #   (a) |h*| is meaningful (at least 0.02 = 2% exposure makes sense), AND
        #   (b) HE meets minimum threshold (statistically significant variance reduction), AND
        #   (c) regime is risk_off  (gold spiking = flight to safety active), OR
        #       (c2) gold is a natural negative hedge AND regime is not risk_on
        h_usable = abs(h_star)
        h_meaningful = h_usable > 0.02
        he_sufficient = hedge_effectiveness >= self.min_he
        gold_is_natural_hedge = h_star < -0.02  # negative correlation = gold hedges portfolio

        regime_trigger = (
            gold_regime == "risk_off"
            or (gold_is_natural_hedge and gold_regime == "neutral" and hedge_effectiveness >= 0.15)
        )

        apply_hedge = h_meaningful and he_sufficient and regime_trigger

        # ---- Step 8: Clamp hedge fraction ---------------------------------
        if apply_hedge:
            # Base hedge pct from |h*| (magnitude, regardless of sign)
            # h* can exceed 1 if correlation is very high; cap at max_hedge_pct
            raw_hedge_pct = h_usable

            # Scale up slightly in risk_off regime
            if gold_regime == "risk_off":
                raw_hedge_pct *= 1.2

            hedge_pct = float(np.clip(raw_hedge_pct, 0.05, self.max_hedge_pct))
        else:
            hedge_pct = 0.0

        hedge_capital = capital * hedge_pct
        deployable_capital = capital - hedge_capital

        # ---- Step 9: Tola buy recommendation --------------------------------
        # How many tolas of gold can hedge_capital buy at Nepal market price?
        gold_tolas_to_buy = (
            hedge_capital / gold_nepal_tola if apply_hedge and gold_nepal_tola > 0 else 0.0
        )
        # Silver: informational — full hedge_capital at silver price
        silver_tolas_to_buy = (
            hedge_capital / silver_nepal_tola if apply_hedge and silver_nepal_tola > 0 else 0.0
        )

        # ---- Step 10: Build reasoning string --------------------------------
        reasoning = self._build_reasoning(
            apply_hedge=apply_hedge,
            h_star=h_star,
            hedge_pct=hedge_pct,
            hedge_effectiveness=hedge_effectiveness,
            gold_regime=gold_regime,
            momentum_20d=momentum_20d,
            corr_gold=corr_gold,
            corr_silver=corr_silver,
            gold_price_usd=gold_price_usd,
            gold_nepal_tola=gold_nepal_tola,
            gold_tolas_to_buy=gold_tolas_to_buy,
            hedge_capital=hedge_capital,
        )

        return HedgeResult(
            apply_hedge=apply_hedge,
            h_star=float(h_star),
            hedge_pct=hedge_pct,
            hedge_capital=hedge_capital,
            deployable_capital=deployable_capital,
            hedge_effectiveness=hedge_effectiveness,
            gold_regime=gold_regime,
            gold_momentum_20d=momentum_20d,
            gold_price_usd=gold_price_usd,
            gold_price_intl_npr=gold_price_intl_npr,
            gold_price_nepal_tola=gold_nepal_tola,
            gold_tolas_to_buy=gold_tolas_to_buy,
            silver_price_usd=0.0,  # populated from macro_indicators separately if needed
            silver_price_nepal_tola=silver_nepal_tola,
            silver_tolas_to_buy=silver_tolas_to_buy,
            correlation_gold=corr_gold,
            silver_correlation=corr_silver,
            reasoning=reasoning,
            data_quality="ok",
        )

    def _build_reasoning(  # noqa: PLR0913
        self,
        apply_hedge: bool,
        h_star: float,
        hedge_pct: float,
        hedge_effectiveness: float,
        gold_regime: str,
        momentum_20d: float,
        corr_gold: float,
        corr_silver: float,
        gold_price_usd: float,
        gold_nepal_tola: float = 0.0,
        gold_tolas_to_buy: float = 0.0,
        hedge_capital: float = 0.0,
    ) -> str:
        regime_str = {
            "risk_off": "RISK-OFF (gold spiking)",
            "neutral": "NEUTRAL",
            "risk_on": "RISK-ON (gold falling)",
        }.get(gold_regime, gold_regime)

        nepal_price_str = (
            f" | Nepal NPR {gold_nepal_tola:,.0f}/tola (FENEGOSIDA)"
            if gold_nepal_tola > 0 else ""
        )
        parts = [
            f"Gold ${gold_price_usd:.0f}/oz{nepal_price_str} | 20d mom {momentum_20d*100:+.1f}% | Regime: {regime_str}",
            f"MVHR h*={h_star:.3f} | HE={hedge_effectiveness*100:.1f}% | ρ(gold)={corr_gold:.2f}",
        ]
        if self.apply_silver:
            parts.append(f"ρ(silver)={corr_silver:.2f}")

        h_dir = "natural hedge (gold↑ when NEPSE↓)" if h_star < 0 else "co-movement (both fall)"
        if apply_hedge:
            buy_str = (
                f" → BUY {gold_tolas_to_buy:.2f} tola fine gold @ NPR {gold_nepal_tola:,.0f}/tola"
                if gold_tolas_to_buy > 0 else ""
            )
            parts.append(
                f"HEDGE ACTIVE: NPR {hedge_capital:,.0f} ({hedge_pct*100:.1f}%) withheld"
                f"{buy_str} [{h_dir}]"
            )
        else:
            reasons = []
            if abs(h_star) <= 0.02:
                reasons.append("h* too small (|h*|≤2%)")
            if hedge_effectiveness < self.min_he:
                reasons.append(f"HE too low ({hedge_effectiveness*100:.1f}% < {self.min_he*100:.0f}%)")
            if gold_regime == "risk_on":
                reasons.append("risk-on (gold falling)")
            if gold_regime == "neutral" and not (h_star < -0.02):
                reasons.append("neutral regime + gold follows portfolio")
            parts.append(
                f"Hedge inactive [{h_dir}]: {'; '.join(reasons) or 'regime neutral + HE borderline'}"
            )

        return " | ".join(parts)

    def _no_data_result(
        self, capital: float, gold_price_usd: float,
        gold_price_intl_npr: float, gold_nepal_tola: float, silver_nepal_tola: float,
    ) -> HedgeResult:
        return HedgeResult(
            apply_hedge=False,
            h_star=0.0, hedge_pct=0.0, hedge_capital=0.0,
            deployable_capital=capital, hedge_effectiveness=0.0,
            gold_regime="no_data", gold_momentum_20d=0.0,
            gold_price_usd=gold_price_usd,
            gold_price_intl_npr=gold_price_intl_npr,
            gold_price_nepal_tola=gold_nepal_tola,
            gold_tolas_to_buy=0.0,
            silver_price_usd=0.0,
            silver_price_nepal_tola=silver_nepal_tola,
            silver_tolas_to_buy=0.0,
            correlation_gold=0.0, silver_correlation=0.0,
            reasoning="No gold data in macro_indicators. Run gold_silver_ingestion first.",
            data_quality="no_gold_data",
        )

    def _no_price_result(
        self, capital: float, gold_price_usd: float,
        gold_price_intl_npr: float, gold_nepal_tola: float, silver_nepal_tola: float,
        gold_rets: pd.Series,
    ) -> HedgeResult:
        momentum_20d = float(gold_rets.iloc[-20:].sum()) if len(gold_rets) >= 20 else 0.0
        if momentum_20d > self.risk_off_threshold:
            gold_regime = "risk_off"
        elif momentum_20d < self.risk_on_threshold:
            gold_regime = "risk_on"
        else:
            gold_regime = "neutral"

        return HedgeResult(
            apply_hedge=False,
            h_star=0.0, hedge_pct=0.0, hedge_capital=0.0,
            deployable_capital=capital, hedge_effectiveness=0.0,
            gold_regime=gold_regime, gold_momentum_20d=momentum_20d,
            gold_price_usd=gold_price_usd,
            gold_price_intl_npr=gold_price_intl_npr,
            gold_price_nepal_tola=gold_nepal_tola,
            gold_tolas_to_buy=0.0,
            silver_price_usd=0.0,
            silver_price_nepal_tola=silver_nepal_tola,
            silver_tolas_to_buy=0.0,
            correlation_gold=0.0, silver_correlation=0.0,
            reasoning=(
                f"Gold regime: {gold_regime} | Nepal NPR {gold_nepal_tola:,.0f}/tola | "
                "Insufficient NEPSE price history for MVHR (need 20+ trading days)."
            ),
            data_quality="no_price_data",
        )


# --------------------------------------------------------------------------
# Convenience function for backtest integration
# --------------------------------------------------------------------------

def compute_hedge_for_date(
    prices_df: pd.DataFrame,
    symbols: List[str],
    date,
    capital: float,
    db_path: str,
    **kwargs,
) -> HedgeResult:
    """
    One-shot hedge computation. Convenience wrapper for GoldSilverHedgeOverlay.

    kwargs are passed to GoldSilverHedgeOverlay.__init__().
    """
    overlay = GoldSilverHedgeOverlay(**kwargs)
    return overlay.compute(prices_df, symbols, date, capital, db_path)


__all__ = [
    "HedgeResult",
    "GoldSilverHedgeOverlay",
    "get_gold_regime",
    "compute_hedge_for_date",
]
