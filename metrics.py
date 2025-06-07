import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class PerformanceMetrics:
    """Calculate trading strategy performance metrics."""
    
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.02):
        """
        Initialize with returns series and risk-free rate.
        
        Args:
            returns (pd.Series): Series of returns
            risk_free_rate (float): Annual risk-free rate (default: 2%)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.annual_factor = 252  # Trading days in a year
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics."""
        return {
            'Total Return': self.total_return(),
            'Annualized Return': self.annualized_return(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown': self.max_drawdown(),
            'Win Rate': self.win_rate(),
            'Profit Factor': self.profit_factor(),
            'Calmar Ratio': self.calmar_ratio()
        }
    
    def total_return(self) -> float:
        """Calculate total return."""
        return (1 + self.returns).prod() - 1
    
    def annualized_return(self) -> float:
        """Calculate annualized return."""
        total_return = self.total_return()
        years = len(self.returns) / self.annual_factor
        return (1 + total_return) ** (1 / years) - 1
    
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = self.returns - self.risk_free_rate / self.annual_factor
        return np.sqrt(self.annual_factor) * excess_returns.mean() / excess_returns.std()
    
    def sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        excess_returns = self.returns - self.risk_free_rate / self.annual_factor
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(self.annual_factor) * excess_returns.mean() / downside_std
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + self.returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()
    
    def win_rate(self) -> float:
        """Calculate win rate."""
        winning_trades = self.returns[self.returns > 0]
        return len(winning_trades) / len(self.returns)
    
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return gains / losses if losses != 0 else float('inf')
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        annualized_return = self.annualized_return()
        max_drawdown = self.max_drawdown()
        return annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate performance metrics for a returns series.
    
    Args:
        returns (pd.Series): Series of returns
        risk_free_rate (float): Annual risk-free rate
        
    Returns:
        Dict[str, float]: Dictionary of performance metrics
    """
    metrics = PerformanceMetrics(returns, risk_free_rate)
    return metrics.calculate_metrics()

if __name__ == "__main__":
    # Example usage
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    metrics = calculate_metrics(returns)
    
    print("\n--- Performance Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
