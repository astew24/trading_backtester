import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(portfolio_df):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df.index, portfolio_df['TotalValue'], label='Portfolio Value')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_signals_with_price(signals_path):
    signals = pd.read_csv(signals_path, index_col='Date', parse_dates=True)
    plt.figure(figsize=(14, 7))
    plt.plot(signals['price'], label='Price')
    plt.plot(signals['SMA_50'], label='SMA 50')
    plt.plot(signals['SMA_200'], label='SMA 200')

    buys = signals[signals['positions'] == 1.0]
    sells = signals[signals['positions'] == -1.0]

    plt.plot(buys.index, buys['price'], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(sells.index, sells['price'], 'v', markersize=10, color='r', label='Sell Signal')

    plt.title('Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()