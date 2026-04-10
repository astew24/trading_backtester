# Liquidity-Aware Equity Strategy Backtester

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

This repo is a small quant research harness for daily-bar equity strategies. The focus is not on maximizing the number of indicators; it is on running short-horizon ideas with realistic trading frictions, benchmark-relative analytics, and a walk-forward workflow that is useful for actual strategy research.

The current setup is strongest for liquid U.S. equities or sector ETFs where you want to test ideas like z-score mean reversion or slower trend-following without pretending daily data can answer microstructure questions it cannot.

## Why this is more than a toy script

- transaction cost model includes commission, spread, and slippage
- exits support stop loss, take profit, and trailing stop logic with conservative daily-bar handling
- mean-reversion signals are liquidity-gated by average dollar volume
- benchmark-aware reporting includes excess return, tracking error, information ratio, beta, and alpha
- walk-forward mode selects parameters on a training window and scores them out of sample
- each run writes a reproducible artifact bundle with config snapshot, metrics, trades, CSV outputs, and charts

## Strategy coverage

- `mean_reversion`: liquidity-aware z-score reversion for short-horizon equity ideas
- `moving_average`: slower crossover baseline for comparison and sanity checks

Both strategies execute on the next bar's open, which keeps the backtest aligned with end-of-day signal generation.

## Repo layout

```text
.
├── src/trading_backtester/   # package code
├── tests/                    # unit tests
├── docs/                     # design notes and limitations
├── config.yaml               # default run + research config
├── main.py                   # local entrypoint without installation
└── Makefile                  # common dev commands
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,app]
```

If you want the lightweight root install path instead:

```bash
pip install -r requirements-dev.txt
```

## Web Demo

The fastest way to show the project to an employer is the Streamlit app. It uses live Yahoo Finance data, the existing moving-average strategy, the existing backtest engine, and read-only in-memory execution.

Run it locally:

```bash
streamlit run streamlit_app.py
```

Deploy it with Streamlit Community Cloud:

1. Push the repo to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the app entrypoint to `streamlit_app.py`.
4. Let Streamlit install dependencies from `requirements.txt`.

## Run a standard backtest

```bash
python3 main.py
```

Override the strategy or symbol universe from the command line:

```bash
python3 main.py --strategy mean_reversion --symbols XLF XLK XLE
```

After installation, the console entrypoint works too:

```bash
trading-backtester --config config.yaml
```

## Run walk-forward validation

The default config includes parameter grids for both strategies. This mode rolls a training window forward, picks the best parameter set on the configured metric, and reports out-of-sample performance.

```bash
python3 main.py --walk-forward
```

Useful knobs in `config.yaml`:

- `research.train_bars`
- `research.test_bars`
- `research.step_bars`
- `research.metric`
- `research.parameter_grid`

## Output artifacts

Each run writes a timestamped directory under `artifacts/` with:

- `results.csv`
- `trades.csv`
- `metrics.json`
- `config_snapshot.yaml`
- `summary.md`
- `benchmark.csv` when a benchmark is configured
- `walk_forward_windows.csv` for walk-forward runs
- performance charts in PNG format

## Development

```bash
make test
make lint
make typecheck
```

There is a short discussion of model assumptions in [docs/assumptions_and_limitations.md](docs/assumptions_and_limitations.md).
