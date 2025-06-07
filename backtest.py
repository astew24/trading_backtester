import pandas as pd
import numpy as np

def run_backtest(signals_path, initial_cash=100000):
    signals = pd.read_csv(signals_path, index_col='Date', parse_dates=True)

    cash = initial_cash
    shares = 0
    portfolio = []

    for date, row in signals.iterrows():
        price = row['price']
        position = row['positions']

        if position == 1.0:  # Buy
            shares = cash // price
            cash -= shares * price
        elif position == -1.0:  # Sell
            cash += shares * price
            shares = 0

        total_value = cash + shares * price
        portfolio.append((date, cash, shares, total_value))

    portfolio_df = pd.DataFrame(portfolio, columns=['Date', 'Cash', 'Shares', 'TotalValue'])
    portfolio_df.set_index('Date', inplace=True)
    return portfolio_df