import pandas as pd
import numpy as np

def generate_signals(data_path, output_path, short_window=50, long_window=200):
    """
    Loads stock data, calculates SMAs, and generates trading signals.
    """
    try:
        # Use index_col='Date' as the primary method
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    except Exception:
        # Fallback for when the 'Date' column isn't a named column
        print("Could not find 'Date' column, attempting to use first column as index.")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Ensure the index is named 'Date'
    data.index.name = 'Date'
    
    # *** THIS IS THE KEY FIX ***
    # Isolate the 'Close' column to ensure we are only working with numbers
    close_prices = data['Close']

    # Create a new DataFrame for the signals, using the same index
    signals = pd.DataFrame(index=data.index)
    signals['price'] = close_prices

    # Calculate SMAs on the clean 'close_prices' series
    signals[f'SMA_{short_window}'] = close_prices.rolling(window=short_window, min_periods=1).mean()
    signals[f'SMA_{long_window}'] = close_prices.rolling(window=long_window, min_periods=1).mean()

    # Generate signal based on SMA crossover
    signals['signal_raw'] = np.where(signals[f'SMA_{short_window}'] > signals[f'SMA_{long_window}'], 1.0, 0.0)

    # Find the exact points of crossover
    signals['positions'] = signals['signal_raw'].diff()

    # Save the signals to a new CSV file
    signals.to_csv(output_path)
    print(f"Signals generated and saved to {output_path}")
    return signals

if __name__ == "__main__":
    input_data_path = 'data/AAPL_data.csv'
    output_data_path = 'data/AAPL_signals.csv'

    # Generate the signals
    trading_signals = generate_signals(input_data_path, output_data_path)

    if trading_signals is not None:
        print("\n--- Last 10 Rows of Data with Signals ---")
        print(trading_signals.tail(10))