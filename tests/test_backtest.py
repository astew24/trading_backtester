import math

import pandas as pd
from trading_backtester.backtest import BacktestConfig, BacktestEngine


def _price_frame(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    close = pd.Series(values, index=index)
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


def test_backtest_realizes_expected_pnl_without_costs() -> None:
    data = _price_frame([100.0, 100.0, 110.0, 110.0])
    signals = pd.DataFrame(
        {"target_position": [1.0, 1.0, 0.0, 0.0]},
        index=data.index,
    )

    engine = BacktestEngine(
        BacktestConfig(
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
    )
    results, trades = engine.run(data, signals, symbol="TEST")

    assert len(trades) == 1
    assert math.isclose(trades[0].gross_pnl, 9500.0, rel_tol=1e-9)
    assert math.isclose(results["Portfolio_Value"].iloc[-1], 109500.0, rel_tol=1e-9)


def test_backtest_applies_transaction_costs_on_entry_and_exit() -> None:
    data = _price_frame([100.0, 100.0, 110.0, 110.0])
    signals = pd.DataFrame(
        {"target_position": [1.0, 1.0, 0.0, 0.0]},
        index=data.index,
    )

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=100000.0,
            position_size=0.95,
            commission_bps=0.0,
            fixed_commission=1.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            stop_loss=None,
            take_profit=None,
            trailing_stop=None,
        )
    )
    results, trades = engine.run(data, signals, symbol="TEST")

    assert len(trades) == 1
    assert math.isclose(trades[0].net_pnl, 9498.0, rel_tol=1e-9)
    assert math.isclose(results["Portfolio_Value"].iloc[-1], 109498.0, rel_tol=1e-9)
