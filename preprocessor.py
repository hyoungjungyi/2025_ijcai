import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from sklearn.linear_model import LinearRegression


class YfinancePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tickers = kwargs.get('tickers', None)  
        self.start_date = kwargs.get('start_date', '2000-01-03')  
        self.end_date = kwargs.get('end_date', '2023-12-29')  
        self.output_path = kwargs.get('output_path', None)  
        self.indicator = kwargs.get('indicator', 'alpha158')  

        self.tickers = [ticker for ticker in self.tickers if ticker != '002030.KS']

    def download_data(self):
        """Yahoo Finance에서 데이터를 다운로드"""
        df_list = []
        for ticker in self.tickers:
            print(f"Downloading data for {ticker}...")

            
            df = yf.download(ticker, start=self.start_date, end=self.end_date, interval="1d", progress=False, auto_adjust=True)
            
            if df.empty:
                print(f"No data for {ticker}")
                continue

            
            df.columns = df.columns.str.lower()

            
            if 'adjclose' not in df.columns:
                df['adjclose'] = df['close']  # adjclose가 없으면 close를 사용

          
            df["ticker"] = ticker
            df["date"] = df.index  

            df_list.append(df)

        if not df_list:
            print("No data was downloaded for any tickers.")
            return pd.DataFrame()

        
        df = pd.concat(df_list, axis=0, ignore_index=True)
        df = df.sort_values(by=["ticker", "date"])
        return df







    def make_feature(self):
        """metric 생성"""
        self.df["zopen"] = self.df["open"] / self.df["close"] - 1
        self.df["zhigh"] = self.df["high"] / self.df["close"] - 1
        self.df["zlow"] = self.df["low"] / self.df["close"] - 1
        self.df["zadjcp"] = self.df["adjclose"] / self.df["close"] - 1
        df_new = self.df.sort_values(by=["ticker", "date"])
        stock = df_new
        unique_ticker = stock.ticker.unique()
        df_indicator = pd.DataFrame()
        
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.ticker == unique_ticker[i]].copy()

            # 최소 데이터 길이 확인
            if len(temp_indicator) < 20:
                print(f"Not enough data for ticker {unique_ticker[i]}, skipping feature generation.")
                continue

            temp_indicator.loc[:, "zclose"] = (temp_indicator["close"] / 
                                            (temp_indicator["close"].rolling(2).sum() - temp_indicator["close"])) - 1
            temp_indicator.loc[:, "zd_5"] = (temp_indicator["close"].rolling(5).mean()) / temp_indicator["close"] - 1
            temp_indicator.loc[:, "zd_10"] = (temp_indicator["close"].rolling(10).mean()) / temp_indicator["close"] - 1
            temp_indicator.loc[:, "zd_15"] = (temp_indicator["close"].rolling(15).mean()) / temp_indicator["close"] - 1
            temp_indicator.loc[:, "zd_20"] = (temp_indicator["close"].rolling(20).mean()) / temp_indicator["close"] - 1
            temp_indicator.loc[:, "zd_25"] = (temp_indicator["close"].rolling(25).mean()) / temp_indicator["close"] - 1
            temp_indicator.loc[:, "zd_30"] = (temp_indicator["close"].rolling(30).mean()) / temp_indicator["close"] - 1
            temp_indicator.loc[:, "zd_60"] = (temp_indicator["close"].rolling(60).mean()) / temp_indicator["close"] - 1
            
            df_indicator = pd.concat([df_indicator, temp_indicator], ignore_index=True)

        
        df_indicator = df_indicator.ffill().bfill()
        
        return df_indicator


    def run(self, custom_data_path=None):
        """데이터 다운로드, 전처리, 특성 생성 및 저장"""
        if not custom_data_path:
            self.df = self.download_data()  
        else:
            self.df = pd.read_csv(custom_data_path)  

        
        self.df = self.df.rename(columns={
            "date": "date",
            "tic": "ticker",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjclose",
            "Volume": "volume",

        })

        self.df = self.make_feature()  #

        if self.output_path:
            self.df.to_csv(self.output_path, index=False)
            print(f"Data saved to {self.output_path}")
        else:
            print("Output path not provided.")



def get_complete_tickers(tickers, start="2000-01-01", end="2023-09-01", output_file=None):
    valid_tickers = []
    all_data = pd.DataFrame()

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, start=start, end=end)
        
        
        if data.empty:
            print(f"No data for {ticker}")
            continue
        
        print(f"Ticker: {ticker}, Data shape: {data.shape}")
        
    
        data_start_year = data.index.min().year
        data_end_year = data.index.max().year

        all_years_present = True
        for year in range(2000, 2024):
            if year not in data.index.year:
                all_years_present = False
                print(f"{ticker} is missing data for year {year}.")
                break

        if all_years_present:
            valid_tickers.append(ticker)
            data['Ticker'] = ticker
            all_data = pd.concat([all_data, data])
        else:
            print(f"{ticker} does not have all data from 2000 to 2023.")

  
    if output_file and not all_data.empty:
        all_data.to_csv(output_file, index=True)
        print(f"Data saved to {output_file}")

    return valid_tickers





file_path_kospi = '/home/hyunjung/projects/2025_ijcai/data/kospi_data.csv'
df_kospi = pd.read_csv(file_path_kospi)
kospi_ticker_list = df_kospi['Issue code'].astype(str).tolist()[:20]
kospi_ticker_list = [ticker + ".KS" for ticker in kospi_ticker_list]


file_path_nasdaq = '/home/hyunjung/projects/2025_ijcai/data/nasdaq_final.csv'
df_nasdaq = pd.read_csv(file_path_nasdaq)
nasdaq_ticker_list = df_nasdaq['Symbol'].astype(str).tolist()[:20]


complete_kospi_tickers = get_complete_tickers(kospi_ticker_list)
complete_nasdaq_tickers = get_complete_tickers(nasdaq_ticker_list)
pd.DataFrame(complete_kospi_tickers, columns=["Ticker"]).to_csv("complete_kospi_tickers.csv", index=False)
pd.DataFrame(complete_nasdaq_tickers, columns=["Ticker"]).to_csv("complete_nasdaq_tickers.csv", index=False)
all_tickers = complete_kospi_tickers + complete_nasdaq_tickers


config = {
    'output_path': 'complete_stock_data.csv',
    'start_date': '2000-01-03',
    'end_date': '2023-12-29',
    'tickers': all_tickers,
    'indicator': 'alpha158'
}


preprocessor = YfinancePreprocessor(**config)
preprocessor.run() 
