# ruff: noqa: I001,E501

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
from trading_backtester.strategies import MeanReversionStrategy, MovingAverageCrossover


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = load_config(BASE_DIR / "config.yaml")
RISK_FREE_RATE = float(DEFAULT_CONFIG["risk"]["risk_free_rate"])
COLOR_SCHEME = {
    "ink": "#102033",
    "slate": "#52606D",
    "sand": "#F6F1EA",
    "cream": "#FBF8F3",
    "line": "#D8CEC1",
    "accent": "#C05A2B",
    "green": "#1F7A5B",
    "red": "#B3412D",
    "gold": "#B88B2E",
}


@st.cache_data(show_spinner=False, ttl=1800)
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
    return f"{value:.2%}" if percent else f"{value:.2f}"


def _format_currency(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(192, 90, 43, 0.12), transparent 32%),
                linear-gradient(180deg, {COLOR_SCHEME["cream"]} 0%, {COLOR_SCHEME["sand"]} 100%);
        }}

        html, body, [class*="css"] {{
            font-family: "IBM Plex Sans", sans-serif;
            color: {COLOR_SCHEME["ink"]};
        }}

        h1, h2, h3, .hero-title {{
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
            color: {COLOR_SCHEME["ink"]};
        }}

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(246, 241, 234, 0.95));
            border-right: 1px solid rgba(16, 32, 51, 0.08);
        }}

        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(16, 32, 51, 0.08);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 18px 36px rgba(16, 32, 51, 0.05);
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLOR_SCHEME["slate"]};
            font-weight: 600;
        }}

        .hero-panel,
        .note-card {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(16, 32, 51, 0.08);
            border-radius: 24px;
            box-shadow: 0 20px 48px rgba(16, 32, 51, 0.06);
            padding: 1.5rem;
        }}

        .eyebrow {{
            color: {COLOR_SCHEME["accent"]};
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            margin-bottom: 0.6rem;
            text-transform: uppercase;
        }}

        .hero-title {{
            font-size: 3rem;
            line-height: 0.95;
            margin: 0 0 0.85rem 0;
        }}

        .hero-copy {{
            color: {COLOR_SCHEME["slate"]};
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }}

        .chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }}

        .chip {{
            background: rgba(16, 32, 51, 0.05);
            border: 1px solid rgba(16, 32, 51, 0.08);
            border-radius: 999px;
            color: {COLOR_SCHEME["ink"]};
            display: inline-block;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.4rem 0.8rem;
        }}

        .note-label {{
            color: {COLOR_SCHEME["slate"]};
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.55rem;
            text-transform: uppercase;
        }}

        .note-value {{
            font-family: "Space Grotesk", sans-serif;
            font-size: 2rem;
            line-height: 1;
            margin-bottom: 0.4rem;
        }}

        .note-copy {{
            color: {COLOR_SCHEME["slate"]};
            font-size: 0.96rem;
            line-height: 1.55;
            margin: 0;
        }}

        .section-title {{
            margin: 0;
        }}

        .subtle-copy {{
            color: {COLOR_SCHEME["slate"]};
            margin-top: 0.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_price_chart(price_data, signals, trades_frame):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["Close"],
            mode="lines",
            name="Close",
            line={"color": COLOR_SCHEME["ink"], "width": 2.6},
        )
    )

    overlay_specs = [
        ("short_ma", "Short MA", COLOR_SCHEME["accent"]),
        ("long_ma", "Long MA", COLOR_SCHEME["green"]),
        ("rolling_mean", "Rolling Mean", COLOR_SCHEME["gold"]),
    ]
    for column, label, color in overlay_specs:
        if column in signals.columns and not signals[column].dropna().empty:
            figure.add_trace(
                go.Scatter(
                    x=signals.index,
                    y=signals[column],
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 2, "dash": "dot"},
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
                marker={
                    "color": COLOR_SCHEME["green"],
                    "size": 9,
                    "symbol": "triangle-up",
                },
            )
        )
        figure.add_trace(
            go.Scatter(
                x=exits["exit_date"],
                y=exits["exit_price"],
                mode="markers",
                name="Exit",
                marker={
                    "color": COLOR_SCHEME["red"],
                    "size": 9,
                    "symbol": "triangle-down",
                },
            )
        )

    figure.update_layout(
        title="Price Action and Executions",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series",
        margin={"l": 18, "r": 18, "t": 56, "b": 18},
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        height=430,
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
            line={"color": COLOR_SCHEME["ink"], "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=benchmark_curve.index,
            y=benchmark_curve,
            mode="lines",
            name="Buy and Hold",
            line={"color": COLOR_SCHEME["slate"], "width": 2, "dash": "dash"},
        )
    )
    figure.update_layout(
        title="Portfolio Value vs Buy and Hold",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        legend_title="Series",
        margin={"l": 18, "r": 18, "t": 56, "b": 18},
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        height=430,
    )
    return figure


def _build_drawdown_chart(results):
    drawdown = (results["Portfolio_Value"] / results["Portfolio_Value"].cummax()) - 1.0
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill="tozeroy",
            line={"color": COLOR_SCHEME["red"], "width": 2},
            name="Drawdown",
        )
    )
    figure.update_layout(
        title="Drawdown Profile",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        margin={"l": 18, "r": 18, "t": 56, "b": 18},
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        height=320,
    )
    return figure


def _build_position_chart(results):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=results.index,
            y=results["Position"],
            mode="lines",
            line={"color": COLOR_SCHEME["accent"], "width": 2.2},
            fill="tozeroy",
            name="Position",
        )
    )
    figure.update_layout(
        title="Net Position Sizing",
        xaxis_title="Date",
        yaxis_title="Net Position",
        margin={"l": 18, "r": 18, "t": 56, "b": 18},
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        height=320,
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
    if "return_pct" in trade_log.columns:
        trade_log["return_pct"] = trade_log["return_pct"].map(
            lambda value: f"{value:.2%}" if math.isfinite(value) else "N/A"
        )
    return trade_log


def _strategy_caption(strategy_key: str) -> str:
    if strategy_key == "moving_average":
        return (
            "Trend-following overlay using a moving-average crossover with "
            "next-bar execution and configurable short exposure."
        )
    return (
        "Liquidity-gated mean reversion using z-scores, conservative exits, "
        "and the same cost model as the research engine."
    )


st.set_page_config(
    page_title="Trading Backtester Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_styles()

today = date.today()
default_end = today
default_start = today - timedelta(days=365 * 2)
backtest_defaults = DEFAULT_CONFIG["backtest"]
strategy_defaults = DEFAULT_CONFIG["strategy"]

with st.sidebar:
    st.header("Research Controls")
    ticker = st.text_input("Ticker", value="QQQ").strip().upper() or "QQQ"
    strategy_key = st.radio(
        "Strategy family",
        options=["moving_average", "mean_reversion"],
        format_func=lambda value: {
            "moving_average": "Trend following",
            "mean_reversion": "Mean reversion",
        }[value],
        horizontal=True,
    )
    st.caption(_strategy_caption(strategy_key))

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)
    initial_capital = st.number_input(
        "Initial capital",
        min_value=1000.0,
        value=float(backtest_defaults["initial_capital"]),
        step=5000.0,
    )

    if strategy_key == "moving_average":
        moving_average_defaults = strategy_defaults["moving_average"]
        short_window = st.slider(
            "Short moving average",
            min_value=5,
            max_value=100,
            value=int(moving_average_defaults["short_window"]),
            step=1,
        )
        long_window = st.slider(
            "Long moving average",
            min_value=20,
            max_value=250,
            value=int(moving_average_defaults["long_window"]),
            step=1,
        )
        allow_short = st.checkbox(
            "Allow shorting",
            value=bool(moving_average_defaults.get("allow_short", True)),
        )
        strategy = MovingAverageCrossover(
            short_window=short_window,
            long_window=long_window,
            allow_short=allow_short,
        )
        minimum_bars = long_window
    else:
        mean_reversion_defaults = strategy_defaults["mean_reversion"]
        lookback = st.slider(
            "Lookback window",
            min_value=5,
            max_value=60,
            value=int(mean_reversion_defaults["lookback"]),
            step=1,
        )
        entry_zscore = st.slider(
            "Entry z-score",
            min_value=0.5,
            max_value=3.0,
            value=float(mean_reversion_defaults["entry_zscore"]),
            step=0.1,
        )
        exit_zscore = st.slider(
            "Exit z-score",
            min_value=0.1,
            max_value=1.5,
            value=float(mean_reversion_defaults["exit_zscore"]),
            step=0.05,
        )
        liquidity_lookback = st.slider(
            "Liquidity lookback",
            min_value=5,
            max_value=60,
            value=int(mean_reversion_defaults["liquidity_lookback"]),
            step=1,
        )
        min_avg_dollar_volume = st.number_input(
            "Minimum avg dollar volume",
            min_value=100000.0,
            value=float(mean_reversion_defaults["min_avg_dollar_volume"]),
            step=500000.0,
        )
        allow_short = st.checkbox(
            "Allow shorting",
            value=bool(mean_reversion_defaults.get("allow_short", True)),
        )
        strategy = MeanReversionStrategy(
            lookback=lookback,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            liquidity_lookback=liquidity_lookback,
            min_avg_dollar_volume=min_avg_dollar_volume,
            allow_short=allow_short,
        )
        minimum_bars = max(lookback, liquidity_lookback)

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

if strategy_key == "moving_average" and short_window >= long_window:
    st.error("Short moving average must be smaller than the long moving average.")
    st.stop()

if strategy_key == "mean_reversion" and exit_zscore >= entry_zscore:
    st.error("Exit z-score must be smaller than the entry z-score.")
    st.stop()

with st.spinner(f"Downloading daily bars for {ticker}..."):
    try:
        price_data = _download_live_data(ticker, start_date, end_date)
    except Exception as exc:
        st.error(f"Unable to load market data for {ticker}: {exc}")
        st.stop()

if len(price_data) < minimum_bars:
    st.error(
        f"Not enough daily bars for the selected settings. "
        f"Fetched {len(price_data)} rows between {start_date} and {end_date}."
    )
    st.stop()

signals = strategy.generate_signals(price_data)
if strategy_key == "mean_reversion":
    signals["rolling_mean"] = (
        price_data["Close"]
        .rolling(strategy.lookback, min_periods=strategy.lookback)
        .mean()
    )

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
    risk_free_rate=RISK_FREE_RATE,
    benchmark_returns=benchmark_curve.pct_change().fillna(0.0),
)
trade_log = _build_trade_log(trades_to_frame(trades))

hero_left, hero_right = st.columns([1.45, 0.85])
with hero_left:
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="eyebrow">Quant Research Demo</div>
            <div class="hero-title">Liquidity-aware strategy backtesting without toy assumptions.</div>
            <p class="hero-copy">
                This app runs the repository's actual engine on live Yahoo Finance daily bars,
                with transaction costs, benchmark-relative analytics, and conservative exit handling.
            </p>
            <div class="chip-row">
                <span class="chip">{ticker}</span>
                <span class="chip">{len(price_data)} daily bars</span>
                <span class="chip">{start_date.isoformat()} to {end_date.isoformat()}</span>
                <span class="chip">Next-bar execution</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown(
        f"""
        <div class="note-card">
            <div class="note-label">Run Snapshot</div>
            <div class="note-value">{_format_currency(float(results["Portfolio_Value"].iloc[-1]))}</div>
            <p class="note-copy">
                Strategy return {_format_ratio(metrics["Total Return"], percent=True)}
                against buy-and-hold {_format_ratio(metrics["Benchmark Return"], percent=True)}.
                Active return is {_format_ratio(metrics["Excess Return"], percent=True)}
                with max drawdown {_format_ratio(metrics["Max Drawdown"], percent=True)}.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Headline Metrics")
metric_row_one = st.columns(4)
metric_row_one[0].metric(
    "Total Return", _format_ratio(metrics["Total Return"], percent=True)
)
metric_row_one[1].metric(
    "Buy and Hold", _format_ratio(metrics["Benchmark Return"], percent=True)
)
metric_row_one[2].metric("Sharpe Ratio", _format_ratio(metrics["Sharpe Ratio"]))
metric_row_one[3].metric(
    "Max Drawdown", _format_ratio(metrics["Max Drawdown"], percent=True)
)

metric_row_two = st.columns(4)
metric_row_two[0].metric(
    "Active Return", _format_ratio(metrics["Excess Return"], percent=True)
)
metric_row_two[1].metric("Win Rate", _format_ratio(metrics["Win Rate"], percent=True))
metric_row_two[2].metric("Profit Factor", _format_ratio(metrics["Profit Factor"]))
metric_row_two[3].metric("Total Trades", f"{int(metrics['Total Trades'])}")

chart_row_one = st.columns(2)
with chart_row_one[0]:
    st.plotly_chart(
        _build_price_chart(price_data, signals, trade_log),
        use_container_width=True,
    )
with chart_row_one[1]:
    st.plotly_chart(
        _build_portfolio_chart(results, benchmark_curve),
        use_container_width=True,
    )

chart_row_two = st.columns(2)
with chart_row_two[0]:
    st.plotly_chart(
        _build_drawdown_chart(results),
        use_container_width=True,
    )
with chart_row_two[1]:
    st.plotly_chart(
        _build_position_chart(results),
        use_container_width=True,
    )

st.markdown("### Trade Review")
action_left, action_right = st.columns([0.72, 0.28])
with action_left:
    st.markdown(
        """
        <h3 class="section-title">Recent Executions</h3>
        <p class="subtle-copy">
            The table shows completed trades after commission, spread, slippage,
            and any stop or profit-taking logic from <code>config.yaml</code>.
        </p>
        """,
        unsafe_allow_html=True,
    )
with action_right:
    st.download_button(
        "Download results CSV",
        results.to_csv().encode("utf-8"),
        file_name=f"{ticker.lower()}_{strategy_key}_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

if trade_log.empty:
    st.info("No completed trades were generated for the selected settings.")
else:
    st.dataframe(
        trade_log.tail(12).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Full Trade Log"):
        st.dataframe(trade_log, use_container_width=True, hide_index=True)

with st.expander("Methodology and Execution Assumptions"):
    st.markdown(
        f"""
        - Data source: live Yahoo Finance daily OHLCV for `{ticker}`
        - Strategy family: `{strategy_key}` using the repository's package code
        - Execution model: next-bar fills with commission, spread, slippage, stop loss,
          take profit, and trailing stop logic from `config.yaml`
        - Benchmark line: buy-and-hold of `{ticker}` over the same period
        - Risk-free rate: `{RISK_FREE_RATE:.2%}` for Sharpe, Sortino, alpha, and information ratio
        - The static project site in `site/` is built from the same engine so the public link
          stays aligned with the actual research code
        """
    )
