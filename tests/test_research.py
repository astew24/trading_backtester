import pandas as pd
from trading_backtester.research import (
    WalkForwardConfig,
    expand_parameter_grid,
    generate_walk_forward_windows,
)


def test_expand_parameter_grid_builds_cartesian_product() -> None:
    parameter_grid = {
        "lookback": [10, 20],
        "entry_zscore": [1.0, 1.5],
        "allow_short": [True],
    }

    candidates = expand_parameter_grid(parameter_grid)

    assert len(candidates) == 4
    assert {"lookback": 10, "entry_zscore": 1.0, "allow_short": True} in candidates
    assert {"lookback": 20, "entry_zscore": 1.5, "allow_short": True} in candidates


def test_generate_walk_forward_windows_uses_rolling_train_and_test_slices() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    config = WalkForwardConfig(train_bars=4, test_bars=2, step_bars=2)

    windows = generate_walk_forward_windows(index, config)

    assert len(windows) == 4
    assert windows[0].train_start == pd.Timestamp("2024-01-01")
    assert windows[0].train_end == pd.Timestamp("2024-01-04")
    assert windows[0].test_start == pd.Timestamp("2024-01-05")
    assert windows[0].test_end == pd.Timestamp("2024-01-06")
    assert windows[-1].test_start == pd.Timestamp("2024-01-11")
    assert windows[-1].test_end == pd.Timestamp("2024-01-12")
