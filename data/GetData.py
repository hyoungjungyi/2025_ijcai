import yfinance as yf
import pandas as pd
import os

# 티커 리스트 불러오기
kospi_tickers = [str(ticker) for ticker in pd.read_csv('data/complete_kospi_tickers.csv')['ticker'].tolist()]
nasdaq_tickers = [str(ticker) for ticker in pd.read_csv('data/complete_nasdaq_tickers.csv')['ticker'].tolist()]

def fetch_and_save_ticker_data(ticker_list, output_csv, start_date='1999-11-01'):
    """
    Fetch stock data for a list of tickers starting from a specified date,
    sort by date and restructure, then save the results to a CSV file.

    Args:
    - ticker_list (list): List of stock tickers
    - output_csv (str): Path to save the resulting CSV file
    - start_date (str): The starting date for the data (format: YYYY-MM-DD)

    Returns:
    - None
    """
    all_data = []

    for ticker in ticker_list:
        print(f"Fetching data for {ticker}...")
        try:
            # Fetch data
            stock_data = yf.Ticker(ticker).history(start=start_date, actions=True)
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
                else:
                    # Default to Close if no dividends are present
                    stock_data['Adj Close'] = stock_data['Close']
            
            # Select and rename required columns
            stock_data = stock_data[['Date', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume']].copy()
            stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'adjclose', 'volume']
            stock_data['ticker'] = ticker  # Add a column for ticker
            
            # Convert 'date' column to desired format and ensure it includes the start_date
            stock_data['date'] = pd.to_datetime(stock_data['date'], utc=True).dt.tz_localize(None)  # UTC -> Localize None
            stock_data = stock_data[stock_data['date'] >= pd.Timestamp(start_date)]  # Filter by start_date
            
            # Append to the list
            all_data.append(stock_data)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if all_data:
        # Concatenate all data into a single DataFrame
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns
        final_df = final_df[['date', 'ticker', 'open', 'close', 'high', 'low', 'adjclose', 'volume']]
        
        # Sort by date
        final_df.sort_values(by=['date', 'ticker'], inplace=True)
        
        # Save to CSV
        final_df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")
    else:
        print("No data to save.")

# Fetch and save KOSPI data /
fetch_and_save_ticker_data(kospi_tickers, "data/raw_kospi_data.csv", start_date='1999-11-01')

# Fetch and save NASDAQ data
fetch_and_save_ticker_data(nasdaq_tickers, "data/raw_nasdaq_data.csv", start_date='1999-11-01')

# Load and display the first 10 rows of each CSV
kospi_data = pd.read_csv("data/raw_kospi_data.csv")
nasdaq_data = pd.read_csv("data/raw_nasdaq_data.csv")

print("KOSPI Data Head(10):")
print(kospi_data.head(10))

print("\nNASDAQ Data Head(10):")
print(nasdaq_data.head(10))
