"""Trading backtester package."""

from .backtest import BacktestConfig, BacktestEngine, Trade
from .metrics import PerformanceMetrics, calculate_metrics
from .research import (
    WalkForwardConfig,
    expand_parameter_grid,
    run_walk_forward_from_config,
)
from .strategies import MeanReversionStrategy, MovingAverageCrossover, build_strategy

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "MeanReversionStrategy",
    "MovingAverageCrossover",
    "PerformanceMetrics",
    "Trade",
    "WalkForwardConfig",
    "build_strategy",
    "calculate_metrics",
    "expand_parameter_grid",
    "run_walk_forward_from_config",
]
