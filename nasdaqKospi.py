import pandas as pd
import yfinance as yf

# KOSPI 데이터 파일 경로 및 티커 목록
file_path = '/home/hyunjung/projects/2025_ijcai/kospi_data.csv'
df = pd.read_csv(file_path)
kospi_ticker_list = df['Issue code'].astype(str).tolist()
kospi_ticker_list = [ticker + ".KS" for ticker in kospi_ticker_list]

# NASDAQ 데이터 파일 경로 및 티커 목록
file_path = '/home/hyunjung/projects/2025_ijcai/nasdaq_final.csv'
df = pd.read_csv(file_path)
nasdaq_ticker_list = df['Symbol'].astype(str).tolist()

complete_kospi_tickers = []
complete_nasdaq_tickers = []

def get_complete_tickers(tickers, start="2000-01-01", end="2023-09-01"):
    valid_tickers = []
    
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            print(f"No data for {ticker}")
            continue
    
        data_start_year = data.index.min().year
        data_end_year = data.index.max().year

        print(f"Start Year for {ticker}: {data_start_year}")
        print(f"End Year for {ticker}: {data_end_year}")
        
        # 시작 년도가 2000년 이하이고 종료 년도가 2023년 이상인 티커만 선택
        if data_start_year <= 2000 and data_end_year >= 2023:
            valid_tickers.append(ticker)
            print(f"{ticker}'s start year {data_start_year} and end year {data_end_year} meet the criteria")
        else:
            print(f"{ticker}'s start year {data_start_year} and end year {data_end_year} do not meet the criteria")

    return valid_tickers


complete_kospi_tickers.extend(get_complete_tickers(kospi_ticker_list))
complete_nasdaq_tickers.extend(get_complete_tickers(nasdaq_ticker_list))

print("\nList of complete kospi tickers:")
print(complete_kospi_tickers)
print("\nList of complete nasdaq tickers:")
print(complete_nasdaq_tickers)


pd.DataFrame(complete_kospi_tickers, columns=["Ticker"]).to_csv("complete_kospi_tickers.csv", index=False)
pd.DataFrame(complete_nasdaq_tickers, columns=["Ticker"]).to_csv("complete_nasdaq_tickers.csv", index=False)
