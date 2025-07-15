from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .data import REQUIRED_PRICE_COLUMNS


def _validate_input_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column for column in REQUIRED_PRICE_COLUMNS if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")
    if data.empty:
        raise ValueError("Input data is empty")
    return data.sort_index().copy()


@dataclass
class BaseStrategy:
    name: str

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class MovingAverageCrossover(BaseStrategy):
    short_window: int = 20
    long_window: int = 100
    allow_short: bool = True

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 100,
        allow_short: bool = True,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        super().__init__(name="moving_average_crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = _validate_input_data(data)
        close = frame["Close"]
        short_ma = close.rolling(
            self.short_window, min_periods=self.short_window
        ).mean()
        long_ma = close.rolling(self.long_window, min_periods=self.long_window).mean()

        target_position = pd.Series(0.0, index=frame.index)
        target_position = target_position.mask(short_ma > long_ma, 1.0)
        if self.allow_short:
            target_position = target_position.mask(short_ma < long_ma, -1.0)

        signals = pd.DataFrame(index=frame.index)
        signals["close"] = close
        signals["short_ma"] = short_ma
        signals["long_ma"] = long_ma
        signals["signal"] = target_position.diff().fillna(target_position)
        signals["target_position"] = target_position.fillna(0.0)
        return signals

    @staticmethod
    def optimize_parameters(
        data: pd.DataFrame,
        *,
        short_windows: range = range(5, 31, 5),
        long_windows: range = range(40, 201, 20),
    ) -> dict[str, float]:
        frame = _validate_input_data(data)
        close = frame["Close"]
        returns = close.pct_change().fillna(0.0)
        best: dict[str, float] | None = None

        for short_window, long_window in product(short_windows, long_windows):
            if short_window >= long_window:
                continue
            strategy = MovingAverageCrossover(
                short_window=short_window, long_window=long_window
            )
            signals = strategy.generate_signals(frame)
            shifted_position = signals["target_position"].shift(1).fillna(0.0)
            strategy_returns = shifted_position * returns
            volatility = strategy_returns.std(ddof=0)
            sharpe = (
                np.sqrt(252) * strategy_returns.mean() / volatility
                if volatility > 0
                else float("-inf")
            )
            result = {
                "best_short_window": float(short_window),
                "best_long_window": float(long_window),
                "best_sharpe_ratio": float(sharpe),
                "best_total_return": float((1 + strategy_returns).prod() - 1),
            }
            if best is None or result["best_sharpe_ratio"] > best["best_sharpe_ratio"]:
                best = result

        if best is None:
            raise ValueError("No valid parameter combinations were evaluated")
        return best


@dataclass
class MeanReversionStrategy(BaseStrategy):
    lookback: int = 20
    entry_zscore: float = 1.5
    exit_zscore: float = 0.25
    liquidity_lookback: int = 20
    min_avg_dollar_volume: float = 5000000.0
    allow_short: bool = True

    def __init__(
        self,
        lookback: int = 20,
        entry_zscore: float = 1.5,
        exit_zscore: float = 0.25,
        liquidity_lookback: int = 20,
        min_avg_dollar_volume: float = 5000000.0,
        allow_short: bool = True,
    ) -> None:
        if exit_zscore >= entry_zscore:
            raise ValueError("exit_zscore must be smaller than entry_zscore")
        super().__init__(name="mean_reversion")
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.liquidity_lookback = liquidity_lookback
        self.min_avg_dollar_volume = min_avg_dollar_volume
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = _validate_input_data(data)
        close = frame["Close"]
        rolling_mean = close.rolling(self.lookback, min_periods=self.lookback).mean()
        rolling_std = close.rolling(self.lookback, min_periods=self.lookback).std(
            ddof=0
        )
        zscore = (close - rolling_mean) / rolling_std.replace(0.0, np.nan)
        avg_dollar_volume = (
            (frame["Close"] * frame["Volume"])
            .rolling(
                self.liquidity_lookback,
                min_periods=self.liquidity_lookback,
            )
            .mean()
        )
        liquid = avg_dollar_volume >= self.min_avg_dollar_volume

        # State machine: iterate bar-by-bar so entry/exit logic is explicit and
        # easy to audit. Vectorized version would be faster but harder to verify.
        positions: list[float] = []
        current_position = 0.0
        for is_liquid, current_zscore in zip(
            liquid.fillna(False), zscore.fillna(0.0), strict=False
        ):
            if not is_liquid:
                current_position = 0.0
            elif current_position == 0.0:
                if current_zscore <= -self.entry_zscore:
                    current_position = 1.0
                elif self.allow_short and current_zscore >= self.entry_zscore:
                    current_position = -1.0
            elif current_position > 0.0 and current_zscore >= -self.exit_zscore:
                current_position = 0.0
            elif current_position < 0.0 and current_zscore <= self.exit_zscore:
                current_position = 0.0
            positions.append(current_position)

        target_position = pd.Series(positions, index=frame.index, dtype=float)
        signals = pd.DataFrame(index=frame.index)
        signals["close"] = close
        signals["zscore"] = zscore
        signals["avg_dollar_volume"] = avg_dollar_volume
        signals["is_liquid"] = liquid.fillna(False)
        signals["signal"] = target_position.diff().fillna(target_position)
        signals["target_position"] = target_position.fillna(0.0)
        return signals


def build_strategy(config: dict[str, Any]) -> BaseStrategy:
    strategy_name = config.get("name", "mean_reversion")
    if strategy_name == "moving_average":
        return MovingAverageCrossover(**config.get("moving_average", {}))
    if strategy_name == "mean_reversion":
        return MeanReversionStrategy(**config.get("mean_reversion", {}))
    raise ValueError(f"Unknown strategy: {strategy_name}")
