"""Long-term portfolio configuration.

6-signal factor model with regime filter:
  Volume Breakout + Quality + Low Volatility + Mean Reversion
  + Quarterly Fundamental + Cross-Sectional Momentum

40 trading-day holds, 5 max positions, regime-adaptive sector limits.
Execution is stock-only: proxy symbols (NEPSE, SECTOR::*) are filtered out.
Run validation/run_all.py after any parameter change to verify performance metrics.
"""

LONG_TERM_CONFIG = {
    "holding_days": 40,
    "max_positions": 5,
    "signal_types": [
        "volume",
        "quality",
        "low_vol",
        "mean_reversion",
        "quarterly_fundamental",  # Point-in-time: PE discount, PB<1+ROE, EPS growth
        "xsec_momentum",          # Cross-sectional 6m-1m momentum (regime-weighted)
    ],
    "rebalance_frequency": 5,
    "stop_loss_pct": 0.08,
    "trailing_stop_pct": 0.10,
    "use_regime_filter": True,
    "sector_limit": 0.35,
    "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 2},
    "bear_threshold": -0.05,
    "initial_capital": 1_000_000,
    # Regime-dependent sector limits
    # Bull: 50% max per sector (trend coherent, concentration helps)
    # Neutral: 35% (balanced diversification)
    # Bear: 25% (tighter diversification to reduce correlated drawdown)
    "regime_sector_limits": {"bull": 0.50, "neutral": 0.35, "bear": 0.25},
    "profit_target_pct": None,
    "event_exit_mode": False,
}
