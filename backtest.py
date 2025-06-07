import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    """Represents a single trade in the backtest."""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position: float  # 1.0 for long, -1.0 for short
    size: float
    pnl: Optional[float] = None
    status: str = "open"  # "open" or "closed"

class Backtest:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0001):  # 0.01% slippage
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.current_capital = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> position size
        self.trades: list[Trade] = []
        self.portfolio_value = []
        self.dates = []
    
    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Run the backtest on the given data and signals.
        
        Args:
            data (pd.DataFrame): Price data
            signals (pd.DataFrame): Trading signals
            
        Returns:
            pd.DataFrame: Backtest results
        """
        self.portfolio_value = [self.initial_capital]
        self.dates = [data.index[0]]
        
        for i in range(1, len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            # Check for position changes
            if 'position' in signals.columns:
                position_change = signals['position'].iloc[i]
                
                if position_change != 0:
                    # Close existing position if any
                    if self.positions:
                        self._close_position(current_date, current_price)
                    
                    # Open new position
                    if position_change > 0:
                        self._open_position(current_date, current_price, 1.0)
                    else:
                        self._open_position(current_date, current_price, -1.0)
            
            # Update portfolio value
            self._update_portfolio_value(current_date, current_price)
        
        return self._generate_results()
    
    def _open_position(self, date: datetime, price: float, direction: float):
        """Open a new position."""
        # Apply slippage
        entry_price = price * (1 + self.slippage * direction)
        
        # Calculate position size (using 95% of available capital)
        position_size = (self.current_capital * 0.95) / entry_price
        
        # Create trade
        trade = Trade(
            entry_date=date,
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            position=direction,
            size=position_size
        )
        
        self.trades.append(trade)
        self.positions['main'] = direction * position_size
        
        # Update capital (subtract commission)
        self.current_capital -= position_size * entry_price * self.commission
    
    def _close_position(self, date: datetime, price: float):
        """Close an existing position."""
        if not self.positions:
            return
        
        # Apply slippage
        exit_price = price * (1 - self.slippage * self.positions['main'])
        
        # Update the last trade
        if self.trades:
            last_trade = self.trades[-1]
            last_trade.exit_date = date
            last_trade.exit_price = exit_price
            last_trade.status = "closed"
            
            # Calculate PnL
            pnl = (exit_price - last_trade.entry_price) * last_trade.size * last_trade.position
            last_trade.pnl = pnl
            
            # Update capital
            self.current_capital += pnl
            self.current_capital -= last_trade.size * exit_price * self.commission
        
        self.positions.clear()
    
    def _update_portfolio_value(self, date: datetime, current_price: float):
        """Update the portfolio value."""
        portfolio_value = self.current_capital
        
        # Add value of open positions
        for position_size in self.positions.values():
            portfolio_value += position_size * current_price
        
        self.portfolio_value.append(portfolio_value)
        self.dates.append(date)
    
    def _generate_results(self) -> pd.DataFrame:
        """Generate backtest results DataFrame."""
        results = pd.DataFrame({
            'Date': self.dates,
            'Portfolio_Value': self.portfolio_value
        })
        results.set_index('Date', inplace=True)
        
        # Calculate returns
        results['Returns'] = results['Portfolio_Value'].pct_change()
        
        return results

def run_backtest(data_path: str, signals_path: str, 
                initial_capital: float = 100000.0) -> pd.DataFrame:
    """
    Run a backtest using the given data and signals.
    
    Args:
        data_path (str): Path to price data
        signals_path (str): Path to signals data
        initial_capital (float): Initial capital for the backtest
        
    Returns:
        pd.DataFrame: Backtest results
    """
    # Load data
    data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    signals = pd.read_csv(signals_path, index_col='Date', parse_dates=True)
    
    # Run backtest
    backtest = Backtest(initial_capital=initial_capital)
    results = backtest.run(data, signals)
    
    return results

if __name__ == "__main__":
    data_path = 'data/AAPL_data.csv'
    signals_path = 'data/AAPL_signals.csv'
    
    results = run_backtest(data_path, signals_path)
    print("\n--- Backtest Results ---")
    print(results.tail())