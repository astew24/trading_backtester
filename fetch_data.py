import yfinance as yf
import pandas as pd
import os

def get_stock_data(ticker, start_date, end_date, output_folder="data"):
    """
    Fetches historical stock data from Yahoo Finance and saves it to a CSV file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print(f"No data found for {ticker} from {start_date} to {end_date}.")
        return None

    file_path = os.path.join(output_folder, f"{ticker}_data.csv")
    data.to_csv(file_path)
    print(f"Data for {ticker} saved to {file_path}")
    return file_path

if __name__ == "__main__":
    ticker_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    get_stock_data(ticker_symbol, start_date, end_date)