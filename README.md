# Trading Backtester

Small research-oriented backtester for daily trading strategies. The project started as a simple moving-average sandbox and has been tightened up into a more reliable workflow with cleaner execution logic, reproducible run artifacts, and basic engineering guardrails.

## What it does

- downloads and caches daily OHLCV data with `yfinance`
- runs either a moving-average crossover strategy or a mean-reversion strategy
- simulates fills with commission, spread, slippage, and stop-based exits
- writes each run to `artifacts/` with results, trades, metrics, config snapshot, and plots
- supports quick local checks with `pytest`, `ruff`, `black`, and `mypy`

## Layout

```text
.
├── src/trading_backtester/   # package code
├── tests/                    # unit tests
├── config.yaml               # default run configuration
├── main.py                   # root entrypoint
├── pyproject.toml            # package + tool configuration
└── Makefile                  # common commands
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run a backtest

Use the default config:

```bash
python main.py
```

Or switch strategy / symbols from the command line:

```bash
python main.py --strategy moving_average --symbols AAPL MSFT
```

You can also run the installed console script after `pip install -e .[dev]`:

```bash
trading-backtester --config config.yaml
```

## Configuration

`config.yaml` controls:

- symbols and date range
- strategy selection and parameters
- transaction cost assumptions
- stop loss / take profit / trailing stop settings
- artifact output behavior

## Outputs

Each run creates a timestamped directory under `artifacts/` containing:

- `results.csv`
- `trades.csv`
- `metrics.json`
- `config_snapshot.yaml`
- `summary.md`
- chart images

There is also a short note on design tradeoffs in `docs/assumptions_and_limitations.md`.

## Development

```bash
make test
make lint
make typecheck
```

If you use pre-commit locally:

```bash
pre-commit install
```
