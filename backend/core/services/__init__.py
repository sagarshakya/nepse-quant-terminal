"""Service façade. Thin, stable, additive-only while the desktop GUI is being built."""
from .market import MarketService
from .signals import SignalService
from .backtests import BacktestService
from .portfolio import PortfolioService

__all__ = ["MarketService", "SignalService", "BacktestService", "PortfolioService"]
