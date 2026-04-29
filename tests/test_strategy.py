import pandas as pd
from trading_backtester.strategies import MeanReversionStrategy, MovingAverageCrossover


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


def test_moving_average_crossover_generates_long_signal_after_cross() -> None:
    data = _price_frame([1.0, 1.0, 1.0, 2.0, 3.0])

    strategy = MovingAverageCrossover(short_window=2, long_window=3, allow_short=False)
    signals = strategy.generate_signals(data)

    assert signals["target_position"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_mean_reversion_enters_on_oversold_and_exits_on_reversion() -> None:
    data = _price_frame([100.0, 100.0, 100.0, 90.0, 95.0, 100.0])

    strategy = MeanReversionStrategy(
        lookback=3,
        entry_zscore=1.0,
        exit_zscore=0.1,
        liquidity_lookback=2,
        min_avg_dollar_volume=1000.0,
        allow_short=False,
    )
    signals = strategy.generate_signals(data)

    assert signals["target_position"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
