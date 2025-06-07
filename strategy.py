import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    position_size: float = 1.0
    pnl: Optional[float] = None

class Strategy:
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
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

class MovingAverageCrossover:
    """Moving Average Crossover strategy implementation."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the strategy with short and long window periods.
        
        Args:
            short_window (int): Short-term moving average window
            long_window (int): Long-term moving average window
        """
        try:
            if short_window >= long_window:
                raise ValueError("Short window must be less than long window")
            self.short_window = short_window
            self.long_window = long_window
            logger.info(f"Initialized MovingAverageCrossover with short_window={short_window}, long_window={long_window}")
        except Exception as e:
            logger.error(f"Error initializing MovingAverageCrossover: {str(e)}")
            raise
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data (pd.DataFrame): DataFrame with 'Close' prices
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        try:
            if 'Close' not in data.columns:
                raise ValueError("DataFrame must contain 'Close' column")
            
            signals = pd.DataFrame(index=data.index)
            signals['Close'] = data['Close']
            
            # Calculate moving averages
            signals['Short_MA'] = signals['Close'].rolling(window=self.short_window, min_periods=1).mean()
            signals['Long_MA'] = signals['Close'].rolling(window=self.long_window, min_periods=1).mean()
            
            # Generate signals
            signals['Signal'] = 0
            signals.loc[signals['Short_MA'] > signals['Long_MA'], 'Signal'] = 1
            signals.loc[signals['Short_MA'] < signals['Long_MA'], 'Signal'] = -1
            
            # Calculate position changes
            signals['Position'] = signals['Signal'].diff()
            
            logger.info(f"Generated signals for {len(signals)} data points")
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000.0) -> Tuple[pd.DataFrame, List[Trade]]:
        """
        Backtest the strategy on historical data.
        
        Args:
            data (pd.DataFrame): DataFrame with 'Close' prices
            initial_capital (float): Initial capital for backtesting
            
        Returns:
            Tuple[pd.DataFrame, List[Trade]]: Results DataFrame and list of trades
        """
        try:
            signals = self.generate_signals(data)
            results = pd.DataFrame(index=signals.index)
            results['Close'] = signals['Close']
            results['Position'] = signals['Position']
            
            # Initialize portfolio
            results['Holdings'] = 0.0
            results['Cash'] = initial_capital
            results['Portfolio_Value'] = initial_capital
            
            # Track trades
            trades = []
            current_trade = None
            
            # Simulate trading
            for i in range(1, len(results)):
                position = results['Position'].iloc[i]
                price = results['Close'].iloc[i]
                
                if position != 0:  # Position change
                    if current_trade is not None:  # Close existing trade
                        current_trade.exit_date = results.index[i]
                        current_trade.exit_price = price
                        current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.position_size
                        trades.append(current_trade)
                        current_trade = None
                    
                    if position > 0:  # Open long position
                        current_trade = Trade(
                            entry_date=results.index[i],
                            entry_price=price,
                            position_size=results['Cash'].iloc[i-1] / price
                        )
                
                # Update portfolio
                if current_trade is not None:
                    results.loc[results.index[i], 'Holdings'] = current_trade.position_size * price
                    results.loc[results.index[i], 'Cash'] = results['Cash'].iloc[i-1]
                else:
                    results.loc[results.index[i], 'Holdings'] = 0.0
                    results.loc[results.index[i], 'Cash'] = results['Cash'].iloc[i-1]
                
                results.loc[results.index[i], 'Portfolio_Value'] = (
                    results['Holdings'].iloc[i] + results['Cash'].iloc[i]
                )
            
            # Close any open trade at the end
            if current_trade is not None:
                current_trade.exit_date = results.index[-1]
                current_trade.exit_price = results['Close'].iloc[-1]
                current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.position_size
                trades.append(current_trade)
            
            logger.info(f"Completed backtest with {len(trades)} trades")
            return results, trades
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise
    
    @staticmethod
    def optimize_parameters(data: pd.DataFrame, 
                          short_windows: List[int] = range(5, 51, 5),
                          long_windows: List[int] = range(20, 201, 10),
                          initial_capital: float = 100000.0) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data (pd.DataFrame): DataFrame with 'Close' prices
            short_windows (List[int]): List of short window periods to test
            long_windows (List[int]): List of long window periods to test
            initial_capital (float): Initial capital for backtesting
            
        Returns:
            Dict: Best parameters and their performance metrics
        """
        try:
            def evaluate_params(params):
                short_window, long_window = params
                if short_window >= long_window:
                    return None
                
                strategy = MovingAverageCrossover(short_window, long_window)
                results, trades = strategy.backtest(data, initial_capital)
                
                # Calculate performance metrics
                returns = results['Portfolio_Value'].pct_change().dropna()
                total_return = (results['Portfolio_Value'].iloc[-1] / initial_capital) - 1
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else float('-inf')
                
                return {
                    'params': (short_window, long_window),
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'num_trades': len(trades)
                }
            
            # Generate parameter combinations
            param_combinations = list(itertools.product(short_windows, long_windows))
            
            # Evaluate parameters in parallel
            with ProcessPoolExecutor() as executor:
                results = list(filter(None, executor.map(evaluate_params, param_combinations)))
            
            if not results:
                raise ValueError("No valid parameter combinations found")
            
            # Find best parameters based on Sharpe ratio
            best_result = max(results, key=lambda x: x['sharpe_ratio'])
            
            logger.info(f"Parameter optimization completed. Best parameters: {best_result['params']}")
            return {
                'best_short_window': best_result['params'][0],
                'best_long_window': best_result['params'][1],
                'best_sharpe_ratio': best_result['sharpe_ratio'],
                'best_total_return': best_result['total_return'],
                'best_num_trades': best_result['num_trades']
            }
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')
    
    # Create strategy instance
    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    print("\n--- Strategy Signals ---")
    print(signals.tail())
    
    # Run backtest
    results, trades = strategy.backtest(data)
    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${results['Portfolio_Value'].iloc[-1]:,.2f}")
    print(f"Number of Trades: {len(trades)}")
    
    # Optimize parameters
    best_params = MovingAverageCrossover.optimize_parameters(data)
    print("\n--- Optimized Parameters ---")
    for key, value in best_params.items():
        print(f"{key}: {value}")