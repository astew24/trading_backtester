from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestConfig, BacktestEngine, Trade
from .config import load_config
from .data import fetch_price_data
from .metrics import calculate_metrics
from .reporting import write_run_artifacts
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
    return parser.parse_args()


def _buy_and_hold_curve(close: pd.Series, initial_capital: float) -> pd.Series:
    returns = close.pct_change().fillna(0.0)
    return initial_capital * (1.0 + returns).cumprod()


def _combine_results(
    results_by_symbol: dict[str, pd.DataFrame], initial_capital: float
) -> pd.DataFrame:
    ordered_results = [
        frame["Returns"].rename(symbol) for symbol, frame in results_by_symbol.items()
    ]
    combined_returns = pd.concat(ordered_results, axis=1).fillna(0.0).mean(axis=1)

    combined = pd.DataFrame(index=combined_returns.index)
    combined["Returns"] = combined_returns
    combined["Portfolio_Value"] = initial_capital * (1.0 + combined_returns).cumprod()
    combined["Cash"] = combined["Portfolio_Value"]
    combined["Shares"] = 0.0
    combined["Market_Value"] = 0.0
    combined["Gross_Exposure"] = (
        pd.concat(
            [
                frame["Gross_Exposure"].rename(symbol)
                for symbol, frame in results_by_symbol.items()
            ],
            axis=1,
        )
        .fillna(0.0)
        .mean(axis=1)
    )
    combined["Net_Exposure"] = (
        pd.concat(
            [
                frame["Net_Exposure"].rename(symbol)
                for symbol, frame in results_by_symbol.items()
            ],
            axis=1,
        )
        .fillna(0.0)
        .mean(axis=1)
    )
    combined["Position"] = combined["Net_Exposure"] / combined[
        "Portfolio_Value"
    ].replace(0.0, pd.NA)
    combined["Turnover"] = (
        pd.concat(
            [
                frame["Turnover"].rename(symbol)
                for symbol, frame in results_by_symbol.items()
            ],
            axis=1,
        )
        .fillna(0.0)
        .mean(axis=1)
    )
    combined["Signal"] = 0.0
    return combined.fillna(0.0)


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
    benchmark_curves: list[pd.Series] = []

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
        benchmark_curves.append(
            _buy_and_hold_curve(price_data["Close"], backtest_config.initial_capital)
        )

    combined_results = _combine_results(
        results_by_symbol, backtest_config.initial_capital
    )
    benchmark = (
        pd.concat(benchmark_curves, axis=1).mean(axis=1) if benchmark_curves else None
    )
    metrics = calculate_metrics(
        combined_results,
        all_trades,
        risk_free_rate=config["risk"].get("risk_free_rate", 0.02),
    )
    report_dir = write_run_artifacts(
        config=config,
        strategy_name=strategy.name,
        symbols=symbols,
        results=combined_results,
        metrics=metrics,
        trades=all_trades,
        benchmark=benchmark,
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

    _, metrics, report_dir = run_from_config(config)
    _print_summary(metrics, report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
