# Trading Strategy Backtester

A Python application for backtesting trading strategies on historical stock data.  
Simulates trades, calculates profitability, evaluates performance metrics, and visualizes results.

## Features

- Download historical stock data (via Yahoo Finance)
- Moving Average Crossover strategy (with parameter optimization)
- Backtesting engine with trade simulation
- Performance metrics: Sharpe, Sortino, Max Drawdown, Win Rate, etc.
- Visualizations: Portfolio value, returns distribution, drawdown, trade analysis
- Configurable via `config.yaml`
- Modular and extensible codebase

## Installation

```sh
git clone https://github.com/astew24/trading_backtester.git
cd trading_backtester
python -m venv venv
source venv/bin/activate  # or .\\venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Usage

Edit `config.yaml` to set your symbols and parameters.

Run the backtester:
```sh
python main.py
```

## Configuration

- `config.yaml` controls:
  - List of stock symbols
  - Backtest period (days)
  - Strategy parameters (initial capital, moving average windows)
  - Risk-free rate

## Project Structure

- `main.py` - Entry point, orchestrates the workflow
- `strategy.py` - Trading strategy logic and parameter optimization
- `metrics.py` - Performance metrics calculations
- `visualize.py` - Plotting and analysis visualizations
- `config.yaml` - User-editable configuration
- `requirements.txt` - Python dependencies

## Example Output

![Portfolio Value Plot](screenshots/portfolio_value.png)
![Returns Distribution](screenshots/returns_distribution.png)

## License

MIT License

---

*Created by [Andrew Stewart](https://github.com/astew24)*
