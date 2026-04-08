"""Trading backtester package."""

from .backtest import BacktestConfig, BacktestEngine, Trade
from .metrics import PerformanceMetrics, calculate_metrics
from .strategies import MeanReversionStrategy, MovingAverageCrossover, build_strategy

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "MeanReversionStrategy",
    "MovingAverageCrossover",
    "PerformanceMetrics",
    "Trade",
    "build_strategy",
    "calculate_metrics",
]
