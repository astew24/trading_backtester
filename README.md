# Trading Strategy Backtester

A robust Python application for backtesting trading strategies against historical stock data. This project provides a framework for testing trading strategies, calculating performance metrics, and visualizing results.

## Features

- Historical data fetching using Yahoo Finance
- Flexible strategy implementation framework
- Realistic backtesting with transaction costs and slippage
- Comprehensive performance metrics calculation
- Interactive visualization of results
- Support for multiple trading strategies

## Project Structure

```
trading_backtester/
├── main.py           # Main script to run backtests
├── strategy.py       # Strategy implementation framework
├── backtest.py       # Backtesting engine
├── metrics.py        # Performance metrics calculation
├── visualize.py      # Results visualization
├── requirements.txt  # Project dependencies
└── data/            # Data storage directory
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading_backtester.git
cd trading_backtester
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run a basic backtest:
```bash
python main.py
```

2. Customize the strategy parameters in `main.py`:
```python
symbol = "AAPL"  # Change the stock symbol
start_date = "2023-01-01"  # Change the start date
end_date = "2023-12-31"    # Change the end date
initial_capital = 100000.0  # Change the initial capital
```

3. Implement your own strategy:
   - Create a new class that inherits from the `Strategy` base class
   - Implement the `generate_signals` method
   - Use your strategy in the backtest

## Performance Metrics

The backtester calculates the following metrics:
- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio

## Visualization

The project includes several visualization tools:
- Portfolio value over time
- Returns distribution
- Drawdown analysis
- Trade analysis
- Performance metrics comparison

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance for providing historical data
- The Python data science community for their excellent libraries
