#2000년부터 2023년까지 데이터 다 있는 티커만 가져오기 => complete_kospi_tickers.csv, complete_nasdaq_tickers.csv
import yfinance as yf
import pandas as pd
import os

def get_complete_tickers(tickers, start="2000-01-01", end="2023-09-01"):
    """모든 기간에 데이터가 있는 티커 필터링"""
    valid_tickers = []
    for ticker in tickers:
        print(f"Validating data for {ticker}...")
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            print(f"No data for {ticker}")
            continue

        all_years_present = all(year in data.index.year for year in range(2000, 2024))
        if all_years_present:
            valid_tickers.append(ticker)
        else:
            print(f"{ticker} does not have complete data from 2000 to 2023.")

    return valid_tickers


# KOSPI 및 NASDAQ 티커 데이터 로드
file_path_kospi = 'data/kospi_ticker.csv'
df_kospi = pd.read_csv(file_path_kospi)
kospi_ticker_list = df_kospi['Ticker'].astype(str).tolist()
kospi_ticker_list = [ticker + ".KS" for ticker in kospi_ticker_list]

file_path_nasdaq = 'data/nasdaq_ticker.csv'
df_nasdaq = pd.read_csv(file_path_nasdaq)
nasdaq_ticker_list = df_nasdaq['Symbol'].astype(str).tolist()


# 유효한 티커 필터링 및 저장
complete_kospi_tickers = get_complete_tickers(kospi_ticker_list)
complete_nasdaq_tickers = get_complete_tickers(nasdaq_ticker_list)

# 결과를 CSV 파일로 저장
pd.DataFrame(complete_kospi_tickers, columns=["ticker"]).to_csv('data/complete_kospi_tickers.csv', index=False)
pd.DataFrame(complete_nasdaq_tickers, columns=["ticker"]).to_csv('data/complete_nasdaq_tickers.csv', index=False)
