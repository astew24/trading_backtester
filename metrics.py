import numpy as np

def calculate_total_return(portfolio_df):
    initial = portfolio_df['TotalValue'].iloc[0]
    final = portfolio_df['TotalValue'].iloc[-1]
    return (final - initial) / initial

def calculate_sharpe_ratio(portfolio_df, risk_free_rate=0.01):
    daily_returns = portfolio_df['TotalValue'].pct_change().dropna()
    excess_returns = daily_returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(portfolio_df):
    cumulative_max = portfolio_df['TotalValue'].cummax()
    drawdown = (portfolio_df['TotalValue'] - cumulative_max) / cumulative_max
    return drawdown.min()
