import yfinance as yf
import pandas as pd
import os

# 티커 리스트 불러오기
csi300_tickers = [str(ticker) for ticker in pd.read_csv('complete_csi300_tickers.csv')['ticker'].tolist()]
dj30_tickers = [str(ticker) for ticker in pd.read_csv('complete_dj30_tickers.csv')['ticker'].tolist()]
kospi_tickers = [str(ticker) for ticker in pd.read_csv('complete_kospi_tickers.csv')['ticker'].tolist()]
nasdaq_tickers = [str(ticker) for ticker in pd.read_csv('complete_nasdaq_tickers.csv')['ticker'].tolist()]
def format_tickers(ticker_list):
    return ", ".join(ticker_list)

print("CSI 300 Tickers:", format_tickers(csi300_tickers))
print("DJ30 Tickers:", format_tickers(dj30_tickers))
print("KOSPI Tickers:", format_tickers(kospi_tickers))
print("NASDAQ Tickers:", format_tickers(nasdaq_tickers))

def fetch_and_save_ticker_data(ticker_list, output_csv, start_date=None, end_date=None):
    """
    Fetch stock data for a list of tickers starting from a specified date,
    sort by date and restructure, then save the results to a CSV file.

    Args:
    - ticker_list (list): List of stock tickers
    - output_csv (str): Path to save the resulting CSV file
    - start_date (str): The starting date for the data (format: YYYY-MM-DD)
    - end_date (str): The ending date for the data (format: YYYY-MM-DD)

    Returns:
    - None
    """
    all_data = []

    for ticker in ticker_list:
        print(f"Fetching data for {ticker}...")
        try:
            # Fetch data
            stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False)
            if stock_data.empty:
                print(f"No data found for {ticker}, skipping...")
                continue

            # Reset index to get 'date' as a column
            stock_data.reset_index(inplace=True)

            # Ensure 'Adj Close' exists
            if 'Adj Close' not in stock_data.columns:
                if 'Dividends' in stock_data.columns and not stock_data['Dividends'].isnull().all():
                    # Calculate Adj Close by subtracting Dividends from Close
                    stock_data['Adj Close'] = stock_data['Close'] - stock_data['Dividends']
                    print('stock adj check (Dividends) -> ', ticker)
                else:
                    # Default to Close if no dividends are present
                    stock_data['Adj Close'] = stock_data['Close']
                    print('stock adj check -> ', ticker)
            # Select and rename required columns
            stock_data = stock_data[['Date', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume']].copy()
            stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'adjclose', 'volume']
            stock_data['tic'] = ticker  # Add a column for ticker

            # Convert 'date' column to desired format
            stock_data['date'] = pd.to_datetime(stock_data['date'], utc=True).dt.tz_localize(
                None)  # UTC -> Localize None

            # Append to the list
            all_data.append(stock_data)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if all_data:
        # Concatenate all data into a single DataFrame
        final_df = pd.concat(all_data, ignore_index=True)

        # Reorder columns
        final_df = final_df[['date', 'tic', 'open', 'close', 'high', 'low', 'adjclose', 'volume']]

        # Sort by date
        final_df.sort_values(by=['date', 'tic'], inplace=True)

        # Save to CSV
        final_df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")
    else:
        print("No data to save.")
# Fetch and save csi300 data
fetch_and_save_ticker_data(csi300_tickers, "raw_csi300_data.csv", start_date='2014-01-01', end_date='2023-12-31') #2014기준 200개 ticker존재

# Fetch and save dj30 data
fetch_and_save_ticker_data(dj30_tickers, "raw_dj30_data.csv", start_date='2000-01-01', end_date='2023-12-31')


# Fetch and save KOSPI data
fetch_and_save_ticker_data(kospi_tickers, "raw_kospi_data.csv", start_date='2000-01-07', end_date='2023-12-31') #1월 7일 상장한 티커 존재

# Fetch and save NASDAQ data
fetch_and_save_ticker_data(nasdaq_tickers, "raw_nasdaq_data.csv", start_date='2000-01-05', end_date='2023-12-31') #1월 5일 상장한 티커존재

# Load and display the first 10 rows of each CSV

csi300_data = pd.read_csv("raw_csi300_data.csv")
dj30_data = pd.read_csv("raw_dj30_data.csv")

print("KOSPI Data Head(10):")
print(csi300_data.head(10))

print("\nNASDAQ Data Head(10):")
print(dj30_data.head(10))


kospi_data = pd.read_csv("raw_kospi_data.csv")
nasdaq_data = pd.read_csv("raw_nasdaq_data.csv")

print("KOSPI Data Head(10):")
print(kospi_data.head(10))

print("\nNASDAQ Data Head(10):")
print(nasdaq_data.head(10))
