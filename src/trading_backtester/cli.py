from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestConfig, BacktestEngine, Trade
from .config import load_config
from .data import fetch_price_data
from .metrics import calculate_metrics
from .portfolio import buy_and_hold_curve, combine_results
from .reporting import write_run_artifacts
from .research import run_walk_forward_from_config
from .strategies import build_strategy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trading backtester")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the YAML config file"
    )
    parser.add_argument("--strategy", help="Override the configured strategy name")
    parser.add_argument("--symbols", nargs="+", help="Override the configured symbols")
    parser.add_argument(
        "--refresh-data", action="store_true", help="Re-download market data"
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation using the research config",
    )
    return parser.parse_args()


def _print_summary(metrics: dict[str, float], report_dir: Path) -> None:
    print("\nBacktest summary")
    print("----------------")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:>24}: {value: .4f}")
        else:
            print(f"{key:>24}: {value}")
    print(f"\nArtifacts written to: {report_dir}")


def run_from_config(
    config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float], Path]:
    strategy = build_strategy(config["strategy"])
    symbols = list(config["data"]["symbols"])
    backtest_config = BacktestConfig.from_dict(config["backtest"])
    engine = BacktestEngine(backtest_config)

    results_by_symbol: dict[str, pd.DataFrame] = {}
    all_trades: list[Trade] = []
    for symbol in symbols:
        price_data = fetch_price_data(
            symbol,
            config["data"].get("start_date"),
            config["data"].get("end_date"),
            interval=config["data"].get("interval", "1d"),
            cache_dir=config["data"].get("cache_dir", "data/raw"),
            refresh=config["data"].get("refresh", False),
        )
        signals = strategy.generate_signals(price_data)
        results, trades = engine.run(price_data, signals, symbol=symbol)
        results_by_symbol[symbol] = results
        all_trades.extend(trades)

    combined_results = combine_results(
        results_by_symbol, backtest_config.initial_capital
    )
    benchmark_symbol = config["data"].get("benchmark")
    benchmark: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    if benchmark_symbol:
        benchmark_data = fetch_price_data(
            benchmark_symbol,
            config["data"].get("start_date"),
            config["data"].get("end_date"),
            interval=config["data"].get("interval", "1d"),
            cache_dir=config["data"].get("cache_dir", "data/raw"),
            refresh=config["data"].get("refresh", False),
        )
        benchmark = (
            buy_and_hold_curve(
                benchmark_data["Close"],
                backtest_config.initial_capital,
            )
            .reindex(combined_results.index)
            .ffill()
        )
        benchmark_returns = benchmark.pct_change().fillna(0.0)
    metrics = calculate_metrics(
        combined_results,
        all_trades,
        risk_free_rate=config["risk"].get("risk_free_rate", 0.02),
        benchmark_returns=benchmark_returns,
    )
    report_dir = write_run_artifacts(
        config=config,
        strategy_name=strategy.name,
        symbols=symbols,
        results=combined_results,
        metrics=metrics,
        trades=all_trades,
        benchmark=benchmark,
        benchmark_label=benchmark_symbol,
    )
    return combined_results, metrics, report_dir


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)

    if args.strategy:
        config["strategy"]["name"] = args.strategy
    if args.symbols:
        config["data"]["symbols"] = args.symbols
    if args.refresh_data:
        config["data"]["refresh"] = True

    if args.walk_forward:
        _, metrics, report_dir = run_walk_forward_from_config(config)
    else:
        _, metrics, report_dir = run_from_config(config)
    _print_summary(metrics, report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
