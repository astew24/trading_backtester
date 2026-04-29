from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_portfolio_value(
    results: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
    title: str = "Portfolio Value",
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(results.index, results["Portfolio_Value"], label="Strategy", linewidth=2)
    if benchmark is not None:
        axis.plot(
            benchmark.index,
            benchmark,
            label=benchmark_label,
            linestyle="--",
            alpha=0.8,
        )
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Value")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_relative_performance(
    results: pd.DataFrame,
    *,
    benchmark: pd.Series,
    benchmark_label: str = "Benchmark",
    title: str = "Relative Performance",
) -> plt.Figure:
    aligned_benchmark = benchmark.reindex(results.index).ffill().dropna()
    strategy = results.loc[aligned_benchmark.index, "Portfolio_Value"]
    relative_series = (strategy / strategy.iloc[0]) - (
        aligned_benchmark / aligned_benchmark.iloc[0]
    )

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(relative_series.index, relative_series.values, color="#dd8452")
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    axis.set_title(f"{title} vs {benchmark_label}")
    axis.set_xlabel("Date")
    axis.set_ylabel("Normalized Excess Return")
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_drawdown(results: pd.DataFrame, *, title: str = "Drawdown") -> plt.Figure:
    portfolio_value = results["Portfolio_Value"]
    rolling_peak = portfolio_value.cummax()
    drawdown = (portfolio_value / rolling_peak) - 1.0
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.fill_between(drawdown.index, drawdown.values, 0.0, color="#c44e52", alpha=0.3)
    axis.plot(drawdown.index, drawdown.values, color="#c44e52", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_returns_distribution(
    results: pd.DataFrame, *, title: str = "Return Distribution"
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.histplot(results["Returns"], bins=40, kde=True, ax=axis, color="#4c72b0")
    axis.set_title(title)
    axis.set_xlabel("Daily Return")
    figure.tight_layout()
    return figure


def plot_trade_pnl(trades: pd.DataFrame, *, title: str = "Trade PnL") -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=trades, x="entry_date", y="net_pnl", ax=axis, color="#55a868")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Trade")
    axis.set_ylabel("Net PnL")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def save_figure(figure: plt.Figure, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_default_charts(
    results: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
    trades: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Path]:
    figures: dict[str, Any] = {
        "portfolio_value": plot_portfolio_value(
            results,
            benchmark=benchmark,
            benchmark_label=benchmark_label,
        ),
        "drawdown": plot_drawdown(results),
        "returns_distribution": plot_returns_distribution(results),
    }
    if benchmark is not None:
        figures["relative_performance"] = plot_relative_performance(
            results,
            benchmark=benchmark,
            benchmark_label=benchmark_label,
        )
    if trades is not None and not trades.empty:
        figures["trade_pnl"] = plot_trade_pnl(trades)

    saved_paths: dict[str, Path] = {}
    for name, figure in figures.items():
        if output_dir is not None:
            saved_paths[name] = save_figure(figure, Path(output_dir) / f"{name}.png")
        elif not show:
            plt.close(figure)
        if show:
            figure.show()

    return saved_paths
