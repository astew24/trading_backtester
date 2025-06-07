import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Calculate trading strategy performance metrics."""
    
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.02):
        """
        Initialize with returns series and risk-free rate.
        
        Args:
            returns (pd.Series): Series of returns
            risk_free_rate (float): Annual risk-free rate (default: 2%)
        """
        try:
            self.returns = pd.to_numeric(returns, errors='coerce').dropna()
            if len(self.returns) == 0:
                raise ValueError("No valid returns data after cleaning")
            self.risk_free_rate = risk_free_rate
            self.annual_factor = 252  # Trading days in a year
            logger.info(f"Initialized PerformanceMetrics with {len(self.returns)} valid returns")
        except Exception as e:
            logger.error(f"Error initializing PerformanceMetrics: {str(e)}")
            raise
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics."""
        try:
            metrics = {
                'Total Return': self.total_return(),
                'Annualized Return': self.annualized_return(),
                'Sharpe Ratio': self.sharpe_ratio(),
                'Sortino Ratio': self.sortino_ratio(),
                'Max Drawdown': self.max_drawdown(),
                'Win Rate': self.win_rate(),
                'Profit Factor': self.profit_factor(),
                'Calmar Ratio': self.calmar_ratio()
            }
            logger.info("Successfully calculated all metrics")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {k: float('nan') for k in ['Total Return', 'Annualized Return', 'Sharpe Ratio', 
                                            'Sortino Ratio', 'Max Drawdown', 'Win Rate', 
                                            'Profit Factor', 'Calmar Ratio']}
    
    def total_return(self) -> float:
        """Calculate total return."""
        try:
            return float((1 + self.returns).prod() - 1)
        except Exception as e:
            logger.error(f"Error calculating total return: {str(e)}")
            return float('nan')
    
    def annualized_return(self) -> float:
        """Calculate annualized return."""
        try:
            total_return = self.total_return()
            years = len(self.returns) / self.annual_factor
            if years <= 0 or (1 + total_return) <= 0:
                return float('nan')
            ann_return = (1 + total_return) ** (1 / years) - 1
            return float(ann_return.real)  # .real in case of complex
        except Exception as e:
            logger.error(f"Error calculating annualized return: {str(e)}")
            return float('nan')
    
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = self.returns - self.risk_free_rate / self.annual_factor
            if excess_returns.std() == 0:
                return float('nan')
            return float(np.sqrt(self.annual_factor) * excess_returns.mean() / excess_returns.std())
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return float('nan')
    
    def sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        try:
            excess_returns = self.returns - self.risk_free_rate / self.annual_factor
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return float('inf')
            downside_std = float(np.sqrt(np.mean(downside_returns ** 2)))
            if downside_std == 0:
                return float('nan')
            return float(np.sqrt(self.annual_factor) * excess_returns.mean() / downside_std)
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return float('nan')
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + self.returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            return float(drawdowns.min())
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return float('nan')
    
    def win_rate(self) -> float:
        """Calculate win rate."""
        try:
            winning_trades = self.returns[self.returns > 0]
            return float(len(winning_trades) / len(self.returns))
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            return float('nan')
    
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        try:
            gains = float(self.returns[self.returns > 0].sum())
            losses = float(abs(self.returns[self.returns < 0].sum()))
            return gains / losses if losses != 0 else float('inf')
        except Exception as e:
            logger.error(f"Error calculating profit factor: {str(e)}")
            return float('nan')
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        try:
            annualized_return = self.annualized_return()
            max_drawdown = self.max_drawdown()
            return float(annualized_return / abs(max_drawdown)) if max_drawdown != 0 else float('inf')
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return float('nan')

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate performance metrics for a returns series.
    
    Args:
        returns (pd.Series): Series of returns
        risk_free_rate (float): Annual risk-free rate
        
    Returns:
        Dict[str, float]: Dictionary of performance metrics
    """
    try:
        metrics = PerformanceMetrics(returns, risk_free_rate)
        return metrics.calculate_metrics()
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {str(e)}")
        return {k: float('nan') for k in ['Total Return', 'Annualized Return', 'Sharpe Ratio', 
                                        'Sortino Ratio', 'Max Drawdown', 'Win Rate', 
                                        'Profit Factor', 'Calmar Ratio']}

if __name__ == "__main__":
    # Example usage
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    metrics = calculate_metrics(returns)
    
    print("\n--- Performance Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
