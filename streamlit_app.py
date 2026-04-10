# ruff: noqa: I001

from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_backtester.backtest import BacktestConfig, BacktestEngine, trades_to_frame
from trading_backtester.config import load_config
from trading_backtester.data import validate_price_data
from trading_backtester.metrics import calculate_metrics
from trading_backtester.strategies import MovingAverageCrossover


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = load_config(BASE_DIR / "config.yaml")


def _download_live_data(ticker: str, start_date: date, end_date: date):
    data = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    return validate_price_data(data)


def _buy_and_hold_curve(price_data, initial_capital: float):
    returns = price_data["Close"].pct_change().fillna(0.0)
    return initial_capital * (1.0 + returns).cumprod()


def _format_ratio(value: float, *, percent: bool = False) -> str:
    if not math.isfinite(value):
        return "N/A"
    if percent:
        return f"{value:.2%}"
    return f"{value:.2f}"


def _build_price_chart(price_data, trades_frame):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["Close"],
            mode="lines",
            name="Close",
            line={"color": "#1f77b4", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["short_ma"],
            mode="lines",
            name="Short MA",
            line={"color": "#ff7f0e", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["long_ma"],
            mode="lines",
            name="Long MA",
            line={"color": "#2ca02c", "width": 2},
        )
    )
    if not trades_frame.empty:
        entries = trades_frame.dropna(subset=["entry_date", "entry_price"])
        exits = trades_frame.dropna(subset=["exit_date", "exit_price"])
        figure.add_trace(
            go.Scatter(
                x=entries["entry_date"],
                y=entries["entry_price"],
                mode="markers",
                name="Entry",
                marker={"color": "#16a34a", "size": 9, "symbol": "triangle-up"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=exits["exit_date"],
                y=exits["exit_price"],
                mode="markers",
                name="Exit",
                marker={"color": "#dc2626", "size": 9, "symbol": "triangle-down"},
            )
        )
    figure.update_layout(
        title="Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
        height=420,
    )
    return figure


def _build_portfolio_chart(results, benchmark_curve):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=results.index,
            y=results["Portfolio_Value"],
            mode="lines",
            name="Strategy",
            line={"color": "#111827", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=benchmark_curve.index,
            y=benchmark_curve,
            mode="lines",
            name="Buy and Hold",
            line={"color": "#94a3b8", "width": 2, "dash": "dash"},
        )
    )
    figure.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        legend_title="Series",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
        height=420,
    )
    return figure


def _build_trade_log(trades_frame):
    if trades_frame.empty:
        return trades_frame
    trade_log = trades_frame.copy()
    for column in ["entry_date", "exit_date"]:
        trade_log[column] = trade_log[column].fillna("").astype(str)
    numeric_columns = [
        "quantity",
        "entry_price",
        "entry_costs",
        "exit_price",
        "exit_costs",
        "gross_pnl",
        "net_pnl",
        "return_pct",
    ]
    for column in numeric_columns:
        if column in trade_log.columns:
            trade_log[column] = trade_log[column].round(4)
    return trade_log


st.set_page_config(
    page_title="Trading Backtester Demo",
    layout="wide",
)

st.title("Trading Backtester Demo")
st.caption(
    "Live Yahoo Finance data, the repo's moving-average strategy, and the existing "
    "backtest engine with transaction-cost assumptions from config.yaml."
)

today = date.today()
default_end = today
default_start = today - timedelta(days=365 * 2)
backtest_defaults = DEFAULT_CONFIG["backtest"]
strategy_defaults = DEFAULT_CONFIG["strategy"]["moving_average"]

with st.sidebar:
    st.header("Backtest Controls")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper() or "AAPL"
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)
    initial_capital = st.number_input(
        "Initial capital",
        min_value=1000.0,
        value=float(backtest_defaults["initial_capital"]),
        step=5000.0,
    )
    short_window = st.slider(
        "Short moving average",
        min_value=5,
        max_value=100,
        value=int(strategy_defaults["short_window"]),
        step=1,
    )
    long_window = st.slider(
        "Long moving average",
        min_value=20,
        max_value=250,
        value=int(strategy_defaults["long_window"]),
        step=1,
    )
    allow_short = st.checkbox(
        "Allow shorting",
        value=bool(strategy_defaults.get("allow_short", True)),
    )

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

if short_window >= long_window:
    st.error("Short moving average must be smaller than the long moving average.")
    st.stop()

with st.spinner(f"Downloading live daily data for {ticker}..."):
    try:
        price_data = _download_live_data(ticker, start_date, end_date)
    except Exception as exc:
        st.error(f"Unable to load market data for {ticker}: {exc}")
        st.stop()

if len(price_data) < long_window:
    st.error(
        f"Not enough daily bars for a {long_window}-day moving average. "
        f"Fetched {len(price_data)} rows between {start_date} and {end_date}."
    )
    st.stop()

strategy = MovingAverageCrossover(
    short_window=short_window,
    long_window=long_window,
    allow_short=allow_short,
)
signals = strategy.generate_signals(price_data)
price_with_signals = price_data.join(signals[["short_ma", "long_ma"]], how="left")

engine = BacktestEngine(
    BacktestConfig(
        **{
            **backtest_defaults,
            "initial_capital": float(initial_capital),
        }
    )
)
results, trades = engine.run(price_data, signals, symbol=ticker)
benchmark_curve = _buy_and_hold_curve(price_data, float(initial_capital))
metrics = calculate_metrics(
    results,
    trades,
    risk_free_rate=float(DEFAULT_CONFIG["risk"]["risk_free_rate"]),
    benchmark_returns=benchmark_curve.pct_change().fillna(0.0),
)
trade_log = _build_trade_log(trades_to_frame(trades))

metric_columns = st.columns(6)
metric_columns[0].metric(
    "Total Return", _format_ratio(metrics["Total Return"], percent=True)
)
metric_columns[1].metric("Sharpe Ratio", _format_ratio(metrics["Sharpe Ratio"]))
metric_columns[2].metric("Sortino Ratio", _format_ratio(metrics["Sortino Ratio"]))
metric_columns[3].metric(
    "Max Drawdown", _format_ratio(metrics["Max Drawdown"], percent=True)
)
metric_columns[4].metric("Win Rate", _format_ratio(metrics["Win Rate"], percent=True))
metric_columns[5].metric("Trades", f"{int(metrics['Total Trades'])}")

chart_left, chart_right = st.columns(2)
with chart_left:
    st.plotly_chart(
        _build_price_chart(price_with_signals, trade_log),
        use_container_width=True,
    )
with chart_right:
    st.plotly_chart(
        _build_portfolio_chart(results, benchmark_curve),
        use_container_width=True,
    )

st.subheader("Trade Log")
if trade_log.empty:
    st.info("No completed trades were generated for the selected settings.")
else:
    st.dataframe(trade_log, use_container_width=True, hide_index=True)

with st.expander("Backtest Details"):
    st.markdown(
        f"""
        - Data source: live Yahoo Finance daily OHLCV for `{ticker}`
        - Signal model: moving-average crossover using `{short_window}` /
          `{long_window}` windows
        - Execution model: next-bar execution with commission, spread, slippage,
          and stop logic from `config.yaml`
        - Benchmark line: buy-and-hold of `{ticker}` over the same period
        """
    )
