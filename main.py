import pandas as pd
from strategy import MovingAverageCrossover
from backtest import Backtest
from metrics import calculate_metrics
from visualize import plot_all_analysis
import yfinance as yf
from datetime import datetime, timedelta

def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for a given symbol.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: Historical price data
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def run_backtest_strategy(symbol: str,
                         start_date: str,
                         end_date: str,
                         short_window: int = 50,
                         long_window: int = 200,
                         initial_capital: float = 100000.0) -> tuple:
    """
    Run a complete backtest for a given symbol and strategy.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        short_window (int): Short moving average window
        long_window (int): Long moving average window
        initial_capital (float): Initial capital for the backtest
        
    Returns:
        tuple: (backtest results, performance metrics, trades)
    """
    # Fetch data
    print(f"Fetching data for {symbol}...")
    data = fetch_data(symbol, start_date, end_date)
    
    # Generate signals
    print("Generating trading signals...")
    strategy = MovingAverageCrossover(short_window, long_window)
    signals = strategy.generate_signals(data)
    
    # Run backtest
    print("Running backtest...")
    backtest = Backtest(initial_capital=initial_capital)
    results = backtest.run(data, signals)
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(results['Returns'])
    
    return results, metrics, backtest.trades

def main():
    # Configuration
    symbol = "AAPL"
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    initial_capital = 100000.0
    
    # Run backtest
    results, metrics, trades = run_backtest_strategy(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # Print results
    print("\n=== Backtest Results ===")
    print(f"\nSymbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    final_value = results['Portfolio_Value'].iloc[-1]
    if hasattr(final_value, 'item'):
        final_value = final_value.item()
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    
    print("\n=== Performance Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_all_analysis(results, metrics, trades)

if __name__ == "__main__":
    main()