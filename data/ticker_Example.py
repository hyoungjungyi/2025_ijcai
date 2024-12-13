import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

# 오류 발생한 티커를 기록할 리스트
error_tickers = []

def fetch_ticker_fast(ticker):
    """Fetch data for a single ticker using yfinance Ticker object."""
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="max")
        if data.empty:
            return ticker, None, None  # No data for this ticker
        start_date = data.index.min()
        end_date = data.index.max()
        return ticker, start_date, end_date
    except Exception as e:
        print(f"Failed to get ticker '{ticker}' reason: {e}")
        error_tickers.append((ticker, str(e)))  # 오류 티커와 메시지 기록
        return ticker, None, None

def get_tickers_with_metadata_fast(tickers, start="2000-01-01", end="2023-09-01", max_workers=10):
    """
    Fetch metadata for multiple tickers using parallel processing for faster results.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_ticker_fast, ticker): ticker for ticker in tickers}
        for future in futures:
            ticker, start_date, end_date = future.result()
            results.append((ticker, start_date, end_date))
    return results

# Load ticker lists (KOSPI & NASDAQ)
file_path_kospi = 'kospi_ticker.csv'
df_kospi = pd.read_csv(file_path_kospi)
kospi_ticker_list = df_kospi['Ticker'].astype(str).tolist()
kospi_ticker_list = [ticker + ".KS" for ticker in kospi_ticker_list]

file_path_nasdaq = 'nasdaq_ticker.csv'
df_nasdaq = pd.read_csv(file_path_nasdaq)
nasdaq_ticker_list = df_nasdaq['Symbol'].astype(str).tolist()

# Use the fast fetching function
kospi_results = get_tickers_with_metadata_fast(kospi_ticker_list)
nasdaq_results = get_tickers_with_metadata_fast(nasdaq_ticker_list)

# Filter based on year and month (2000-01 to 2023-12)
def is_complete(start_date, end_date):
    if start_date and end_date:
        return (
            start_date.year < 2000 or (start_date.year == 2000 and start_date.month <= 1)
        ) and (
            end_date.year > 2023 or (end_date.year == 2023 and end_date.month >= 12)
        )
    return False

kospi_complete = [(ticker, start, end) for ticker, start, end in kospi_results if is_complete(start, end)]
nasdaq_complete = [(ticker, start, end) for ticker, start, end in nasdaq_results if is_complete(start, end)]

incomplete_kospi = [(ticker, start, end) for ticker, start, end in kospi_results if not is_complete(start, end)]
incomplete_nasdaq = [(ticker, start, end) for ticker, start, end in nasdaq_results if not is_complete(start, end)]

# Save results to CSV files
pd.DataFrame(kospi_complete, columns=["ticker", "start_date", "end_date"]).to_csv('complete_kospi_tickers.csv', index=False)
pd.DataFrame(nasdaq_complete, columns=["ticker", "start_date", "end_date"]).to_csv('complete_nasdaq_tickers.csv', index=False)

pd.DataFrame(incomplete_kospi, columns=["ticker", "start_date", "end_date"]).to_csv('incomplete_kospi_tickers.csv', index=False)
pd.DataFrame(incomplete_nasdaq, columns=["ticker", "start_date", "end_date"]).to_csv('incomplete_nasdaq_tickers.csv', index=False)

# Save error tickers to a CSV file
pd.DataFrame(error_tickers, columns=["ticker", "error_message"]).to_csv('error_tickers.csv', index=False)
