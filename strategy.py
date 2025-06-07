import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from the input data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has the required columns.
        
        Args:
            data (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)

class MovingAverageCrossover(Strategy):
    """Moving Average Crossover Strategy."""
    
    def __init__(self, short_window: int = 50, long_window: int = 200):
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        
        # Calculate moving averages
        signals[f'SMA_{self.short_window}'] = data['Close'].rolling(
            window=self.short_window, min_periods=1).mean()
        signals[f'SMA_{self.long_window}'] = data['Close'].rolling(
            window=self.long_window, min_periods=1).mean()
        
        # Generate signals
        signals['signal'] = 0.0
        signals['signal'] = np.where(
            signals[f'SMA_{self.short_window}'] > signals[f'SMA_{self.long_window}'], 
            1.0, 0.0
        )
        
        # Generate positions
        signals['position'] = signals['signal'].diff()
        
        return signals

def run_strategy(data_path: str, output_path: str, 
                short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
    """
    Run the moving average crossover strategy on the given data.
    
    Args:
        data_path (str): Path to input data file
        output_path (str): Path to save output signals
        short_window (int): Short moving average window
        long_window (int): Long moving average window
        
    Returns:
        pd.DataFrame: DataFrame with signals
    """
    try:
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    except Exception:
        print("Could not find 'Date' column, attempting to use first column as index.")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    data.index.name = 'Date'
    
    strategy = MovingAverageCrossover(short_window, long_window)
    signals = strategy.generate_signals(data)
    
    signals.to_csv(output_path)
    print(f"Signals generated and saved to {output_path}")
    return signals

if __name__ == "__main__":
    input_data_path = 'data/AAPL_data.csv'
    output_data_path = 'data/AAPL_signals.csv'
    
    signals = run_strategy(input_data_path, output_data_path)
    
    if signals is not None:
        print("\n--- Last 10 Rows of Data with Signals ---")
        print(signals.tail(10))