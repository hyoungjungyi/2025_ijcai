import os
import pandas as pd
import gspread
import numpy as np
import yfinance as yf
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import time
today= datetime.today().date()
# 🔹 Google Sheets API 인증 설정
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials/deep_chronos_google_api.json", scope)
client = gspread.authorize(creds)


# 🔹 Google Sheets 문서 URL 리스트 (여러 개 사용 가능)
URLS = [
    "https://docs.google.com/spreadsheets/d/1bRkd6crVHwwZes4bGBT1CzxOVGBo9C38g9aOwiZM5mE/edit#gid=0",
    "https://docs.google.com/spreadsheets/d/1cB5Mxyzabc123456789/edit#gid=0",
    "https://docs.google.com/spreadsheets/d/1dE5Fjklmnop987654321/edit#gid=0"
]
spreadsheet_index = 0  # ✅ Google Sheets 문서 인덱스 초기화
spreadsheet = client.open_by_url(URLS[spreadsheet_index])

worksheets = {
    "market": spreadsheet.worksheet("market"),
    "results": spreadsheet.worksheet("result"),
    "past": spreadsheet.worksheet("market_past")
}

# 🔹 Google Sheets 데이터 업로드 함수 (셀 한도 초과 시 자동 분할 저장)
def upload_to_gsheets(df, worksheet, batch_size=1000, max_retries=3):
    """ 데이터를 Google Sheets에 업로드 (셀 한도 초과 시 새로운 문서로 분할 저장) """
    global spreadsheet_index
    worksheet.clear()
    time.sleep(2)  # API 요청 속도를 조절하기 위해 대기
    
    worksheet.append_row(df.columns.tolist())  # 컬럼명 업로드
    time.sleep(2)

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].values.tolist()
        retries = 0
        while retries < max_retries:
            try:
                worksheet.append_rows(batch)
                print(f"✅ Uploaded rows {i} to {i + len(batch)}")
                time.sleep(2)
                break
            except gspread.exceptions.APIError as e:
                if "above the limit of 10000000 cells" in str(e):
                    print(f"⚠️ Google Sheets cell limit reached. Switching to next document.")

                    # 다음 Google Sheets 문서로 변경
                    spreadsheet_index += 1
                    if spreadsheet_index >= len(URLS):
                        print("❌ No more Google Sheets available. Stopping upload.")
                        return
                    
                    spreadsheet = client.open_by_url(URLS[spreadsheet_index])
                    worksheet = spreadsheet.worksheet("result")
                    worksheet.clear()
                    time.sleep(2)
                    worksheet.append_row(df.columns.tolist())
                    time.sleep(2)
                    retries = 0  # 새로운 문서로 재시작
                else:
                    print(f"⚠️ API Error (attempt {retries + 1}): {e}")
                    retries += 1
                    time.sleep(5)
                    if retries == max_retries:
                        print(f"❌ Failed to upload rows {i} to {i + len(batch)} after {max_retries} attempts")

# 🔹 Yahoo Finance 데이터 크롤링 함수
def fetch_stock_data(ticker, market):
    """ 특정 티커의 야후파이낸스 데이터를 가져오는 함수 """
    try:
        stock = yf.Ticker(ticker)
        
        # ✅ Step 1: 1일치 데이터 가져오기
        df = stock.history(period="1d", auto_adjust=False)

        # ✅ Step 2: 오늘 데이터가 없으면, 최근 5일치 데이터 가져오기
        if df.empty:
            print(f"⚠️ No data found for {ticker} today. Trying last 5 days...")
            df = stock.history(period="5d", auto_adjust=False)
            df = df.tail(1)  # 가장 최근 거래일 데이터 선택

        # ✅ Step 3: 5일치 데이터도 없으면, 상장폐지로 간주하고 넘어감
        if df.empty:
            print(f"❌ No data found for {ticker} in the last 5 days. Possibly delisted.")
            return None

        # ✅ Step 4: 데이터 정리
        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "date",
            "Open": "open",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        }, inplace=True)

        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["close"]
        df.rename(columns={"Adj Close": "adjclose"}, inplace=True)

        df["ticker"] = ticker
        df["market"] = market
        df["zadjcp"] = df["adjclose"] / df["close"]

        return df[["date", "ticker", "market", "open", "close", "high", "low", "adjclose", "volume", "zadjcp"]]

    except Exception as e:
        print(f"⚠️ Error fetching data for {ticker}: {e}")
        return None


# 🔹 티커 목록 불러오기
tickers_folder = "/hdd1/hyunjung/2025_ijcai/data"
tickers_files = [f for f in os.listdir(tickers_folder) if f.startswith("complete_") and f.endswith("_tickers.csv")]

if tickers_files:
    tickers_dfs = []
    for f in tickers_files:
        market_name = f.split("_")[1]
        df = pd.read_csv(os.path.join(tickers_folder, f))
        
        if "ticker" in df.columns:
            df = df[["ticker"]].copy()
            df["market"] = market_name
            tickers_dfs.append(df)

    if tickers_dfs:
        tickers_df = pd.concat(tickers_dfs, ignore_index=True)
        print("✅ Ticker Data Loaded")
        
        
    else:
        print("❌ No valid ticker column found in CSV files.")
        tickers_df = None
else:
    print("❌ No matching ticker CSV files found.")
    tickers_df = None

# 🔹 Yahoo Finance 데이터 크롤링 및 Google Sheets 업로드
#if tickers_df is not None:
    #all_stock_data = []

    #for _, row in tickers_df.iterrows():
        #stock_df = fetch_stock_data(row["ticker"], row["market"])
        #if stock_df is not None:
            #all_stock_data.append(stock_df)

    #if all_stock_data:
        #final_stock_df = pd.concat(all_stock_data, ignore_index=True)
        #print("✅ Yahoo Finance Data Fetched")

        #final_stock_df["date"] = final_stock_df["date"].astype(str)

        #✅ Google Sheets 업로드 (batch_size 추가)
        #upload_to_gsheets(final_stock_df, worksheets["market"], batch_size=1000)
        #print("✅ Yahoo Finance Data Uploaded to Google Sheets")
    #else:
        #print("❌ No stock data fetched from Yahoo Finance")

def upload_all_demo_results():
    """ demo_results 폴더 내 모든 CSV 파일을 Google Sheets 'results' 시트에 업로드 """
    demo_folder_path ="/hdd1/hyunjung/2025_ijcai/demo_results"
    if not os.path.exists(demo_folder_path):
        print("❌ demo_results 폴더가 존재하지 않습니다.")
        return

    csv_files = [f for f in os.listdir(demo_folder_path) if f.endswith(".csv")]
    if not csv_files:
        print("❌ demo_results 폴더에 CSV 파일이 없습니다.")
        return
    
    all_dataframes = []

    for csv_file in csv_files:
        file_path = os.path.join(demo_folder_path, csv_file)
        
        # ✅ 파일명에서 모델명 & 시장 정보 추출
        model, market, _ = csv_file.split("_")
        print(f"📤 Uploading {csv_file} to Google Sheets ('results')...")

        df = pd.read_csv(file_path)
        all_dataframes.append(df)
    # ✅ Google Sheets 업로드
    final_df = pd.concat(all_dataframes, ignore_index=True)
    upload_to_gsheets(final_df, worksheets["results"], batch_size=1000)
    print(f"✅ {csv_file} 업로드 완료!")
    print("✅ Portfolio Summary Data Uploaded to Google Sheets (results)")

upload_all_demo_results()




# 🔹 상위 5개 티커 선택 (시장별)
def get_top_5_highest_price_tickers(df):
    """ 시장별로 가장 비싼 주식 5개 선택 """
    latest_data = []

    for market in df["market"].unique():
        market_df = df[df["market"] == market]

        # ✅ 시장별로 가장 높은 가격(adjclose 기준) 상위 5개 선택
        top_5_tickers = market_df.sort_values(by="adjclose", ascending=False).head(5)["ticker"].tolist()
        latest_data.extend([(market, ticker) for ticker in top_5_tickers])

    return latest_data
# ✅ market 시트에서 데이터 불러오기
market_data = pd.DataFrame(worksheets["market"].get_all_records())
# ✅ 시장별 상위 5개 티커 선택
top_5_tickers_per_market = get_top_5_highest_price_tickers(market_data)
# 🔹 상위 5개 티커의 과거 2개월치 데이터 가져오기
def fetch_past_stock_data(ticker, market):
    """ 특정 티커의 지난 2개월치 야후파이낸스 데이터를 가져오는 함수 """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo",interval="1d",auto_adjust=False)  # 2개월치 데이터

        if df.empty:
            print(f"⚠️ No past data found for {ticker}. Skipping...")
            return None

        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "date",
            "Open": "open",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        }, inplace=True)

        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["close"]
        df.rename(columns={"Adj Close": "adjclose"}, inplace=True)

        df["ticker"] = ticker
        df["market"] = market
        df["zadjcp"] = df["adjclose"] / df["close"]

        return df[["date", "ticker", "market", "open", "close", "high", "low", "adjclose", "volume", "zadjcp"]]

    except Exception as e:
        print(f"⚠️ Error fetching past data for {ticker}: {e}")
        return None
# 🔹 과거 데이터 크롤링 및 Google Sheets 업로드
if top_5_tickers_per_market:
    all_past_stock_data = []

    for market, ticker in top_5_tickers_per_market:
        past_stock_df = fetch_past_stock_data(ticker, market)
        if past_stock_df is not None:
            all_past_stock_data.append(past_stock_df)

    if all_past_stock_data:
        final_past_stock_df = pd.concat(all_past_stock_data, ignore_index=True)
        print("✅ Past Yahoo Finance Data Fetched")

        final_past_stock_df["date"] = final_past_stock_df["date"].astype(str)

        # ✅ Google Sheets `"market_past"` 시트에 업로드
        upload_to_gsheets(final_past_stock_df, worksheets["past"], batch_size=1000)
        print("✅ Past Stock Data Uploaded to Google Sheets (market_past)")
    else:
        print("❌ No past stock data fetched from Yahoo Finance")
