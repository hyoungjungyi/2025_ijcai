import pandas as pd
import yfinance as yf

# KOSPI 데이터 파일 경로 및 티커 목록
file_path = '/home/hyunjung/projects/2025_ijcai/data/kospi_data.csv'
df = pd.read_csv(file_path)
kospi_ticker_list = df['Issue code'].astype(str).tolist()[:3]  # 상위 20개의 티커만 사용
kospi_ticker_list = [ticker + ".KS" for ticker in kospi_ticker_list]  # KOSPI 종목은 .KS 접미사 추가

# NASDAQ 데이터 파일 경로 및 티커 목록
file_path = '/home/hyunjung/projects/2025_ijcai/data/nasdaq_final.csv'
df = pd.read_csv(file_path)
nasdaq_ticker_list = df['Symbol'].astype(str).tolist()[:3]  # 상위 20개의 티커만 사용

# 데이터를 저장할 경로
output_dir = '/home/hyunjung/projects/2025_ijcai/'

def download_and_save_data(tickers, exchange, start="2000-01-01", end="2023-12-31"):
    for ticker in tickers:
        print(f"Downloading data for {ticker} on {exchange} exchange...")
        data = yf.download(ticker, start=start, end=end)
        if not data.empty:
            # 데이터가 있는 경우 CSV 파일로 저장
            file_name = f"{output_dir}{ticker}_{exchange}.csv"
            data.to_csv(file_name)
            print(f"Saved data for {ticker} to {file_name}")
        else:
            print(f"No data for {ticker}")

# 나스닥 및 코스피 데이터를 각각 다운로드하고 저장
download_and_save_data(nasdaq_ticker_list, "NASDAQ")
download_and_save_data(kospi_ticker_list, "KOSPI")
