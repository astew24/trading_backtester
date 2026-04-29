from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "symbols": ["AAPL", "MSFT", "GOOG"],
        "benchmark": "SPY",
        "start_date": "2020-01-01",
        "end_date": None,
        "interval": "1d",
        "refresh": False,
        "cache_dir": "data/raw",
    },
    "strategy": {
        "name": "mean_reversion",
        "moving_average": {
            "short_window": 20,
            "long_window": 100,
            "allow_short": True,
        },
        "mean_reversion": {
            "lookback": 20,
            "entry_zscore": 1.5,
            "exit_zscore": 0.25,
            "liquidity_lookback": 20,
            "min_avg_dollar_volume": 5000000,
            "allow_short": True,
        },
    },
    "backtest": {
        "initial_capital": 100000.0,
        "position_size": 0.95,
        "max_leverage": 1.0,
        "commission_bps": 5.0,
        "fixed_commission": 1.0,
        "spread_bps": 2.0,
        "slippage_bps": 1.0,
        "volatility_target": None,
        "volatility_lookback": 20,
        "stop_loss": 0.08,
        "take_profit": 0.15,
        "trailing_stop": 0.1,
        "close_positions_on_finish": True,
    },
    "reporting": {
        "output_dir": "artifacts",
        "save_plots": True,
        "show_plots": False,
        "save_trades": True,
    },
    "risk": {
        "risk_free_rate": 0.02,
    },
    "research": {
        "train_bars": 252,
        "test_bars": 63,
        "step_bars": 63,
        "metric": "Sharpe Ratio",
        "parameter_grid": {
            "mean_reversion": {
                "lookback": [10, 20, 30],
                "entry_zscore": [1.0, 1.5, 2.0],
                "exit_zscore": [0.25, 0.5],
                "allow_short": [True],
            },
            "moving_average": {
                "short_window": [10, 20, 30],
                "long_window": [50, 100, 150],
                "allow_short": [True],
            },
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    return deep_merge(DEFAULT_CONFIG, user_config)
