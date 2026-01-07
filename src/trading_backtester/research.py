from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from json import dumps
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestConfig, BacktestEngine, Trade
from .config import deep_merge
from .data import fetch_price_data
from .metrics import calculate_metrics
from .portfolio import buy_and_hold_curve, combine_results, rebuild_results_from_ratios
from .reporting import write_run_artifacts
from .strategies import build_strategy


@dataclass(frozen=True)
class WalkForwardWindow:
    """A single fold in a rolling walk-forward validation run."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardConfig:
    train_bars: int = 252
    test_bars: int = 63
    step_bars: int = 63
    metric: str = "Sharpe Ratio"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> WalkForwardConfig:
        return cls(
            train_bars=int(config.get("train_bars", 252)),
            test_bars=int(config.get("test_bars", 63)),
            step_bars=int(config.get("step_bars", config.get("test_bars", 63))),
            metric=str(config.get("metric", "Sharpe Ratio")),
        )


def expand_parameter_grid(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not parameter_grid:
        return [{}]

    keys = list(parameter_grid)
    values = [
        value if isinstance(value, list) else [value]
        for value in parameter_grid.values()
    ]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def generate_walk_forward_windows(
    index: pd.Index,
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    if config.train_bars <= 0 or config.test_bars <= 0 or config.step_bars <= 0:
        raise ValueError("train_bars, test_bars, and step_bars must be positive")

    windows: list[WalkForwardWindow] = []
    if len(index) < config.train_bars + config.test_bars:
        return windows

    cursor = config.train_bars
    while cursor + config.test_bars <= len(index):
        train_index = index[cursor - config.train_bars : cursor]
        test_index = index[cursor : cursor + config.test_bars]
        windows.append(
            WalkForwardWindow(
                train_start=pd.Timestamp(train_index[0]),
                train_end=pd.Timestamp(train_index[-1]),
                test_start=pd.Timestamp(test_index[0]),
                test_end=pd.Timestamp(test_index[-1]),
            )
        )
        cursor += config.step_bars
    return windows


def _load_price_data_by_symbol(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    price_data_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in config["data"]["symbols"]:
        price_data_by_symbol[symbol] = fetch_price_data(
            symbol,
            config["data"].get("start_date"),
            config["data"].get("end_date"),
            interval=config["data"].get("interval", "1d"),
            cache_dir=config["data"].get("cache_dir", "data/raw"),
            refresh=config["data"].get("refresh", False),
        )
    return price_data_by_symbol


def _common_index(frames: dict[str, pd.DataFrame]) -> pd.Index:
    iterator = iter(frames.values())
    first = next(iterator)
    shared_index = first.index
    for frame in iterator:
        shared_index = shared_index.intersection(frame.index)
    return shared_index


def _strategy_config_with_params(
    strategy_config: dict[str, Any],
    strategy_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(strategy_config)
    merged[strategy_name] = deep_merge(merged.get(strategy_name, {}), params)
    return merged


def _run_symbol_backtest(
    *,
    data: pd.DataFrame,
    strategy_config: dict[str, Any],
    backtest_config: BacktestConfig,
    symbol: str,
    live_start: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[Trade]]:
    strategy = build_strategy(strategy_config)
    signals = strategy.generate_signals(data)
    if live_start is not None:
        warmup_mask = signals.index < live_start
        if warmup_mask.any():
            signals.loc[warmup_mask, "target_position"] = 0.0
            if "signal" in signals.columns:
                signals.loc[warmup_mask, "signal"] = 0.0
    engine = BacktestEngine(backtest_config)
    return engine.run(data, signals, symbol=symbol)


def _evaluate_window(
    *,
    price_data_by_symbol: dict[str, pd.DataFrame],
    strategy_config: dict[str, Any],
    backtest_config: BacktestConfig,
    risk_free_rate: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    benchmark_returns: pd.Series | None = None,
    warmup_start: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, float], list[Trade]]:
    results_by_symbol: dict[str, pd.DataFrame] = {}
    all_trades: list[Trade] = []

    for symbol, full_data in price_data_by_symbol.items():
        window_start = warmup_start if warmup_start is not None else start
        window_data = full_data.loc[window_start:end]
        results, trades = _run_symbol_backtest(
            data=window_data,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            symbol=symbol,
            live_start=start if warmup_start is not None else None,
        )
        live_results = results.loc[start:end]
        results_by_symbol[symbol] = live_results
        all_trades.extend(
            [
                trade
                for trade in trades
                if trade.entry_date >= start and trade.entry_date <= end
            ]
        )

    combined = combine_results(
        results_by_symbol=results_by_symbol,
        initial_capital=backtest_config.initial_capital,
    )
    benchmark_slice = (
        benchmark_returns.loc[start:end] if benchmark_returns is not None else None
    )
    metrics = calculate_metrics(
        combined,
        all_trades,
        risk_free_rate=risk_free_rate,
        benchmark_returns=benchmark_slice,
    )
    return combined, metrics, all_trades


def _window_feature_frame(results: pd.DataFrame) -> pd.DataFrame:
    feature_frame = pd.DataFrame(index=results.index)
    portfolio_value = results["Portfolio_Value"].replace(0.0, pd.NA)
    feature_frame["Returns"] = results["Returns"].fillna(0.0)
    feature_frame["Gross_Ratio"] = (results["Gross_Exposure"] / portfolio_value).fillna(
        0.0
    )
    feature_frame["Net_Ratio"] = (results["Net_Exposure"] / portfolio_value).fillna(0.0)
    feature_frame["Turnover"] = results["Turnover"].fillna(0.0)
    return feature_frame.fillna(0.0)


def _build_oos_results(
    window_results: list[pd.DataFrame],
    initial_capital: float,
) -> pd.DataFrame:
    feature_frame = (
        pd.concat([_window_feature_frame(frame) for frame in window_results])
        .sort_index()
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
    )
    return rebuild_results_from_ratios(
        returns=feature_frame["Returns"],
        gross_exposure_ratio=feature_frame["Gross_Ratio"],
        net_exposure_ratio=feature_frame["Net_Ratio"],
        turnover=feature_frame["Turnover"],
        initial_capital=initial_capital,
    )


def run_walk_forward_from_config(
    config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float], Path]:
    research_config = WalkForwardConfig.from_dict(config.get("research", {}))
    strategy_name = config["strategy"]["name"]
    parameter_grid = config["research"]["parameter_grid"].get(strategy_name, {})
    candidates = expand_parameter_grid(parameter_grid)
    price_data_by_symbol = _load_price_data_by_symbol(config)
    common_index = _common_index(price_data_by_symbol)
    windows = generate_walk_forward_windows(common_index, research_config)

    if not windows:
        raise ValueError("Not enough overlapping history to run walk-forward analysis")
    if not candidates:
        raise ValueError(f"No parameter candidates configured for {strategy_name}")

    benchmark_symbol = config["data"].get("benchmark")
    benchmark_curve: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    if benchmark_symbol:
        benchmark_data = fetch_price_data(
            benchmark_symbol,
            config["data"].get("start_date"),
            config["data"].get("end_date"),
            interval=config["data"].get("interval", "1d"),
            cache_dir=config["data"].get("cache_dir", "data/raw"),
            refresh=config["data"].get("refresh", False),
        ).loc[common_index]
        benchmark_curve = buy_and_hold_curve(
            benchmark_data["Close"],
            config["backtest"]["initial_capital"],
        )
        benchmark_returns = benchmark_curve.pct_change().fillna(0.0)

    backtest_config = BacktestConfig.from_dict(config["backtest"])
    risk_free_rate = config["risk"].get("risk_free_rate", 0.02)
    window_records: list[dict[str, Any]] = []
    oos_windows: list[pd.DataFrame] = []
    oos_trades: list[Trade] = []

    for window in windows:
        best_params: dict[str, Any] | None = None
        best_metric = float("-inf")

        for params in candidates:
            strategy_config = _strategy_config_with_params(
                config["strategy"],
                strategy_name,
                params,
            )
            _, train_metrics, _ = _evaluate_window(
                price_data_by_symbol=price_data_by_symbol,
                strategy_config=strategy_config,
                backtest_config=backtest_config,
                risk_free_rate=risk_free_rate,
                start=window.train_start,
                end=window.train_end,
                benchmark_returns=benchmark_returns,
            )
            score = float(train_metrics.get(research_config.metric, float("-inf")))
            if score > best_metric:
                best_metric = score
                best_params = params

        if best_params is None:
            raise ValueError("Failed to select walk-forward parameters")

        selected_strategy_config = _strategy_config_with_params(
            config["strategy"],
            strategy_name,
            best_params,
        )
        test_results, test_metrics, test_trades = _evaluate_window(
            price_data_by_symbol=price_data_by_symbol,
            strategy_config=selected_strategy_config,
            backtest_config=backtest_config,
            risk_free_rate=risk_free_rate,
            start=window.test_start,
            end=window.test_end,
            benchmark_returns=benchmark_returns,
            warmup_start=window.train_start,
        )
        oos_windows.append(test_results)
        oos_trades.extend(test_trades)
        window_records.append(
            {
                "train_start": window.train_start.date().isoformat(),
                "train_end": window.train_end.date().isoformat(),
                "test_start": window.test_start.date().isoformat(),
                "test_end": window.test_end.date().isoformat(),
                "selected_parameters": dumps(best_params, sort_keys=True),
                "train_metric": best_metric,
                "test_total_return": test_metrics.get("Total Return", float("nan")),
                "test_sharpe_ratio": test_metrics.get("Sharpe Ratio", float("nan")),
                "test_max_drawdown": test_metrics.get("Max Drawdown", float("nan")),
            }
        )

    oos_results = _build_oos_results(
        window_results=oos_windows,
        initial_capital=backtest_config.initial_capital,
    )
    aligned_benchmark_curve = (
        benchmark_curve.reindex(oos_results.index).ffill()
        if benchmark_curve is not None
        else None
    )
    aligned_benchmark_returns = (
        aligned_benchmark_curve.pct_change().fillna(0.0)
        if aligned_benchmark_curve is not None
        else None
    )
    oos_metrics = calculate_metrics(
        oos_results,
        oos_trades,
        risk_free_rate=risk_free_rate,
        benchmark_returns=aligned_benchmark_returns,
    )
    extra_summary_sections = [
        "",
        "## Walk-Forward Validation",
        "",
        f"- Windows: {len(window_records)}",
        f"- Train Bars: {research_config.train_bars}",
        f"- Test Bars: {research_config.test_bars}",
        f"- Step Bars: {research_config.step_bars}",
        f"- Selection Metric: {research_config.metric}",
    ]
    report_dir = write_run_artifacts(
        config=config,
        strategy_name=f"{strategy_name}_walk_forward",
        symbols=list(config["data"]["symbols"]),
        results=oos_results,
        metrics=oos_metrics,
        trades=oos_trades,
        benchmark=aligned_benchmark_curve,
        benchmark_label=benchmark_symbol,
        extra_summary_sections=extra_summary_sections,
    )
    pd.DataFrame(window_records).to_csv(
        report_dir / "walk_forward_windows.csv", index=False
    )
    with (report_dir / "research_config.json").open("w", encoding="utf-8") as handle:
        handle.write(dumps(config["research"], indent=2, sort_keys=True))
        handle.write("\n")
    return oos_results, oos_metrics, report_dir
