from fetch_data import get_stock_data
from strategy import generate_signals
from backtest import run_backtest
from metrics import calculate_total_return, calculate_sharpe_ratio, calculate_max_drawdown
from visualize import plot_equity_curve, plot_signals_with_price

def main():
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    data_path = get_stock_data(ticker, start_date, end_date)

    signal_path = 'data/AAPL_signals.csv'
    generate_signals(data_path, signal_path)

    portfolio_df = run_backtest(signal_path)
    print("Total Return:", calculate_total_return(portfolio_df))
    print("Sharpe Ratio:", calculate_sharpe_ratio(portfolio_df))
    print("Max Drawdown:", calculate_max_drawdown(portfolio_df))

    plot_equity_curve(portfolio_df)
    plot_signals_with_price(signal_path)

if __name__ == "__main__":
    main()