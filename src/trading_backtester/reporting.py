from __future__ import annotations

from datetime import datetime
from json import dumps
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .backtest import Trade, trades_to_frame
from .visualize import render_default_charts


def create_run_id(strategy_name: str, symbols: list[str]) -> str:
    """Build a timestamped run ID from strategy name and up to 3 symbols."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    symbol_slug = "-".join(symbols[:3]).lower()
    return f"{timestamp}_{strategy_name}_{symbol_slug}"


def write_run_artifacts(
    *,
    config: dict[str, Any],
    strategy_name: str,
    symbols: list[str],
    results: pd.DataFrame,
    metrics: dict[str, float],
    trades: list[Trade],
    benchmark: pd.Series | None = None,
    benchmark_label: str | None = None,
    extra_summary_sections: list[str] | None = None,
) -> Path:
    output_root = Path(config["reporting"]["output_dir"])
    run_id = create_run_id(strategy_name, symbols)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(run_dir / "results.csv")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        handle.write(dumps(metrics, indent=2))
        handle.write("\n")

    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    trade_frame = trades_to_frame(trades)
    if config["reporting"].get("save_trades", True):
        trade_frame.to_csv(run_dir / "trades.csv", index=False)

    if benchmark is not None:
        benchmark.rename(benchmark_label or "benchmark").to_csv(
            run_dir / "benchmark.csv"
        )

    render_default_charts(
        results,
        benchmark=benchmark,
        benchmark_label=benchmark_label or "Benchmark",
        trades=trade_frame if not trade_frame.empty else None,
        output_dir=run_dir if config["reporting"].get("save_plots", True) else None,
        show=config["reporting"].get("show_plots", False),
    )

    n_trades = len(trade_frame)
    raw_summary_lines: list[str | None] = [
        f"# Backtest Run: {strategy_name}",
        "",
        f"Symbols: {', '.join(symbols)}",
        (
            f"Benchmark: {benchmark_label}"
            if benchmark is not None and benchmark_label
            else None
        ),
        f"Rows: {len(results)}",
        f"Total trades: {n_trades}",
        "",
        "## Headline Metrics",
        "",
    ]
    summary_lines = [line for line in raw_summary_lines if line is not None]
    for key, value in metrics.items():
        if isinstance(value, float):
            summary_lines.append(f"- {key}: {value:.4f}")
        else:
            summary_lines.append(f"- {key}: {value}")
    if extra_summary_sections:
        summary_lines.extend(extra_summary_sections)

    with (run_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))
        handle.write("\n")

    return run_dir
