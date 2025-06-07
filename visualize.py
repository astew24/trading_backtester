import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

def plot_portfolio_value(results: pd.DataFrame, 
                        title: str = "Portfolio Value Over Time",
                        figsize: tuple = (12, 6)) -> None:
    """
    Plot portfolio value over time.
    
    Args:
        results (pd.DataFrame): DataFrame with portfolio values
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(results.index, results['Portfolio_Value'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_returns_distribution(returns: pd.Series,
                            title: str = "Returns Distribution",
                            figsize: tuple = (10, 6)) -> None:
    """
    Plot returns distribution with normal distribution overlay.
    
    Args:
        returns (pd.Series): Series of returns
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    sns.histplot(returns, kde=True)
    
    # Add normal distribution overlay
    x = np.linspace(returns.min(), returns.max(), 100)
    mu = returns.mean()
    sigma = returns.std()
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    
    plt.title(title)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(results: pd.DataFrame,
                 title: str = "Portfolio Drawdown",
                 figsize: tuple = (12, 6)) -> None:
    """
    Plot portfolio drawdown over time.
    
    Args:
        results (pd.DataFrame): DataFrame with portfolio values
        title (str): Plot title
        figsize (tuple): Figure size
    """
    # Calculate drawdown
    portfolio_value = results['Portfolio_Value']
    rolling_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    
    plt.figure(figsize=figsize)
    plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown.values, color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_metrics(metrics: Dict[str, float],
                title: str = "Performance Metrics",
                figsize: tuple = (10, 6)) -> None:
    """
    Plot performance metrics as a bar chart.
    
    Args:
        metrics (Dict[str, float]): Dictionary of performance metrics
        title (str): Plot title
        figsize (tuple): Figure size
    """
    # Filter out infinite values
    metrics = {k: v for k, v in metrics.items() if not np.isinf(v)}
    
    plt.figure(figsize=figsize)
    plt.bar(metrics.keys(), metrics.values())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_trade_analysis(trades: list,
                       title: str = "Trade Analysis",
                       figsize: tuple = (12, 8)) -> None:
    """
    Plot trade analysis including entry/exit points and PnL.
    
    Args:
        trades (list): List of Trade objects
        title (str): Plot title
        figsize (tuple): Figure size
    """
    if not trades:
        print("No trades to analyze")
        return
    
    # Extract trade data
    entry_dates = [t.entry_date for t in trades]
    entry_prices = [t.entry_price for t in trades]
    exit_dates = [t.exit_date for t in trades if t.exit_date]
    exit_prices = [t.exit_price for t in trades if t.exit_price]
    pnls = [t.pnl for t in trades if t.pnl is not None]
    
    plt.figure(figsize=figsize)
    
    # Plot entry points
    plt.scatter(entry_dates, entry_prices, color='green', marker='^', label='Entry')
    
    # Plot exit points
    plt.scatter(exit_dates, exit_prices, color='red', marker='v', label='Exit')
    
    # Connect entry and exit points
    for i in range(len(entry_dates)):
        if i < len(exit_dates):
            plt.plot([entry_dates[i], exit_dates[i]], 
                    [entry_prices[i], exit_prices[i]], 
                    'k--', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_all_analysis(results: pd.DataFrame,
                     metrics: Dict[str, float],
                     trades: Optional[list] = None) -> None:
    """
    Generate all analysis plots.
    
    Args:
        results (pd.DataFrame): Backtest results
        metrics (Dict[str, float]): Performance metrics
        trades (list, optional): List of Trade objects
    """
    plot_portfolio_value(results)
    plot_returns_distribution(results['Returns'])
    plot_drawdown(results)
    plot_metrics(metrics)
    if trades:
        plot_trade_analysis(trades)

if __name__ == "__main__":
    # Example usage
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    portfolio_values = np.random.normal(100000, 1000, len(dates)).cumsum()
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    
    results = pd.DataFrame({
        'Portfolio_Value': portfolio_values,
        'Returns': returns
    }, index=dates)
    
    metrics = {
        'Sharpe Ratio': 1.5,
        'Sortino Ratio': 2.0,
        'Max Drawdown': -0.15,
        'Win Rate': 0.6
    }
    
    plot_all_analysis(results, metrics)