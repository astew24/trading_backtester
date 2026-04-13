import pandas as pd
from trading_backtester.backtest import BacktestConfig
from trading_backtester.research import (
    WalkForwardConfig,
    _run_symbol_backtest,
    expand_parameter_grid,
    generate_walk_forward_windows,
)


def _price_frame(close_values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(close_values), freq="D")
    close = pd.Series(close_values, index=index)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
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


def test_run_symbol_backtest_preserves_last_warmup_signal_for_first_live_bar() -> None:
    data = _price_frame([100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0])
    strategy_config = {
        "name": "moving_average",
        "moving_average": {
            "short_window": 2,
            "long_window": 3,
            "allow_short": False,
        },
    }
    backtest_config = BacktestConfig(
        initial_capital=100000.0,
        position_size=0.95,
        commission_bps=0.0,
        fixed_commission=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
    )

    results, trades = _run_symbol_backtest(
        data=data,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        symbol="TEST",
        live_start=pd.Timestamp("2024-01-06"),
    )

    assert results.loc[:"2024-01-05", "Shares"].eq(0.0).all()
    assert results.loc["2024-01-06", "Shares"] > 0.0
    assert len(trades) == 1
    assert trades[0].entry_date == pd.Timestamp("2024-01-06")
