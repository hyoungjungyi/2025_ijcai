import yfinance as yf
import pandas as pd


dow_tickers = ['AAPL', 'MSFT', 'JNJ', 'V', 'UNH', 'WMT', 'PG', 'XOM', 'JPM', 'CVX', 
               'HD', 'PFE', 'KO', 'MCD', 'AMGN', 'MRK', 'CSCO', 'VZ', 'IBM', 'MMM', 
               'CAT', 'NKE', 'DIS', 'TRV', 'GS', 'AXP', 'BA', 'INTC', 'WBA', 'DOW']

start_date = "2000-01-01"
end_date = "2023-12-31"
valid_tickers = []


for ticker in dow_tickers:
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print(f"No data for {ticker}")
        continue
    

    data_start = data.index.min()
    data_end = data.index.max()


    print(f"Data Start Date for {ticker}: {data_start}")
    print(f"Data End Date for {ticker}: {data_end}")
    

    if data_start <= pd.Timestamp("2000-01-03") and data_end >= pd.Timestamp("2023-12-29"):
        valid_tickers.append(ticker)
        print(f"{ticker}'s start {data_start} and end {data_end} thus meets the criteria")
    else:
        print(f"{ticker}s start {data_start} and end {data_end} thus does not meet the criteria")


print("Valid tickers with start before 2000 and end after 2023:")
print(valid_tickers)
