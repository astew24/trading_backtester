# ruff: noqa: E402,E501,I001

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trading_backtester.backtest import BacktestConfig, BacktestEngine, trades_to_frame
from trading_backtester.data import validate_price_data
from trading_backtester.metrics import calculate_metrics
from trading_backtester.strategies import MeanReversionStrategy, MovingAverageCrossover


OUTPUT_PATH = ROOT / "site" / "data" / "demo-data.json"
LOOKBACK_YEARS = 4


@dataclass(frozen=True)
class Scenario:
    slug: str
    label: str
    ticker: str
    strategy_family: str
    description: str
    bullets: list[str]
    parameters: dict[str, Any]


SCENARIOS = [
    Scenario(
        slug="qqq-trend",
        label="QQQ Trend",
        ticker="QQQ",
        strategy_family="moving_average",
        description=(
            "Low-turnover long-only trend overlay on Nasdaq exposure. This is the clean, "
            "easy-to-explain baseline for the backtest engine."
        ),
        bullets=[
            "Moving-average crossover with the engine's next-open execution rule.",
            "Long-only configuration keeps the narrative realistic for a public demo.",
            "Useful as a benchmark-friendly example with modest turnover.",
        ],
        parameters={
            "short_window": 20,
            "long_window": 100,
            "allow_short": False,
        },
    ),
    Scenario(
        slug="uso-mean-reversion",
        label="USO Mean Reversion",
        ticker="USO",
        strategy_family="mean_reversion",
        description=(
            "Liquidity-gated long-only mean reversion on an energy ETF. This run shows the "
            "engine doing more than simple trend overlays."
        ),
        bullets=[
            "Entry and exit use rolling z-scores rather than raw price thresholds.",
            "Liquidity filter suppresses trades when average dollar volume drops below the threshold.",
            "Trade count is materially higher than the trend examples, which helps stress frictions.",
        ],
        parameters={
            "lookback": 20,
            "entry_zscore": 1.5,
            "exit_zscore": 0.25,
            "liquidity_lookback": 20,
            "min_avg_dollar_volume": 5_000_000.0,
            "allow_short": False,
        },
    ),
    Scenario(
        slug="meta-trend",
        label="META Trend",
        ticker="META",
        strategy_family="moving_average",
        description=(
            "A higher-beta single-name trend example. This scenario is intentionally more volatile, "
            "but it demonstrates that the engine can still capture strong directional regimes."
        ),
        bullets=[
            "Same execution model as the ETF examples, so the comparison stays apples-to-apples.",
            "Higher return comes with a visibly larger drawdown profile.",
            "Good interview demo for discussing benchmark-relative performance versus raw CAGR.",
        ],
        parameters={
            "short_window": 20,
            "long_window": 100,
            "allow_short": False,
        },
    ),
]


def _download_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    return validate_price_data(data)


def _make_strategy(scenario: Scenario):
    if scenario.strategy_family == "moving_average":
        return MovingAverageCrossover(**scenario.parameters)
    return MeanReversionStrategy(**scenario.parameters)


def _clean_number(value: Any, digits: int = 6) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _series_payload(dates: pd.Index, values: pd.Series) -> list[float | None]:
    series = pd.to_numeric(values, errors="coerce")
    return [_clean_number(value) for value in series.reindex(dates)]


def _trade_markers(trade_frame: pd.DataFrame, date_column: str, price_column: str):
    if trade_frame.empty:
        return []
    frame = trade_frame.dropna(subset=[date_column, price_column]).copy()
    return [
        {
            "date": str(row[date_column])[:10],
            "price": _clean_number(row[price_column], digits=4),
        }
        for _, row in frame.iterrows()
    ]


def _recent_trades(trade_frame: pd.DataFrame) -> list[dict[str, Any]]:
    if trade_frame.empty:
        return []
    columns = [
        "entry_date",
        "exit_date",
        "side",
        "net_pnl",
        "return_pct",
        "exit_reason",
    ]
    frame = trade_frame[columns].copy().tail(8).iloc[::-1]
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        records.append(
            {
                "entry_date": str(row["entry_date"])[:10],
                "exit_date": (
                    str(row["exit_date"])[:10] if pd.notna(row["exit_date"]) else ""
                ),
                "side": str(row["side"]).title(),
                "net_pnl": _clean_number(row["net_pnl"], digits=2),
                "return_pct": _clean_number(row["return_pct"], digits=4),
                "exit_reason": str(row["exit_reason"]).replace("_", " ").title(),
            }
        )
    return records


def build_payload() -> dict[str, Any]:
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS)
    backtest_config = BacktestConfig()
    scenario_payloads: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        price_data = _download_data(scenario.ticker, start_date, end_date)
        strategy = _make_strategy(scenario)
        signals = strategy.generate_signals(price_data)
        if scenario.strategy_family == "mean_reversion":
            signals["rolling_mean"] = (
                price_data["Close"]
                .rolling(
                    scenario.parameters["lookback"],
                    min_periods=scenario.parameters["lookback"],
                )
                .mean()
            )

        results, trades = BacktestEngine(backtest_config).run(
            price_data,
            signals,
            symbol=scenario.ticker,
        )
        benchmark_curve = (
            backtest_config.initial_capital
            * (1.0 + price_data["Close"].pct_change().fillna(0.0)).cumprod()
        )
        metrics = calculate_metrics(
            results,
            trades,
            benchmark_returns=price_data["Close"].pct_change().fillna(0.0),
        )
        trade_frame = trades_to_frame(trades)

        overlays = []
        overlay_specs = [
            ("short_ma", "Short MA", "#C05A2B"),
            ("long_ma", "Long MA", "#1F7A5B"),
            ("rolling_mean", "Rolling Mean", "#B88B2E"),
        ]
        for column, label, color in overlay_specs:
            if column in signals.columns and not signals[column].dropna().empty:
                overlays.append(
                    {
                        "label": label,
                        "color": color,
                        "values": _series_payload(price_data.index, signals[column]),
                    }
                )

        scenario_payloads.append(
            {
                **asdict(scenario),
                "period": {
                    "start": str(price_data.index[0])[:10],
                    "end": str(price_data.index[-1])[:10],
                    "bars": int(len(price_data)),
                },
                "metrics": {
                    key: _clean_number(value, digits=6)
                    for key, value in metrics.items()
                },
                "price": {
                    "dates": [str(timestamp)[:10] for timestamp in price_data.index],
                    "close": _series_payload(price_data.index, price_data["Close"]),
                    "overlays": overlays,
                    "entries": _trade_markers(trade_frame, "entry_date", "entry_price"),
                    "exits": _trade_markers(trade_frame, "exit_date", "exit_price"),
                },
                "equity": {
                    "dates": [str(timestamp)[:10] for timestamp in results.index],
                    "strategy": _series_payload(
                        results.index, results["Portfolio_Value"]
                    ),
                    "benchmark": _series_payload(results.index, benchmark_curve),
                },
                "recent_trades": _recent_trades(trade_frame),
            }
        )

    return {
        "generated_at": str(date.today()),
        "repo_url": "https://github.com/astew24/trading_backtester",
        "scenarios": scenario_payloads,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
