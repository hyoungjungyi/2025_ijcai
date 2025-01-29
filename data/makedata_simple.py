import yfinance as yf
import numpy as np
import pandas as pd
import os
import time


# 티커 유효성 검증 함수 정의
def validate_tickers(tickers):
    valid_tickers = []
    invalid_tickers = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            # 'regularMarketPrice'가 존재하면 유효한 티커로 간주
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception:
            invalid_tickers.append(ticker)
    if invalid_tickers:
        print(f"유효하지 않은 티커 발견 및 제외: {invalid_tickers[:5]} ...")  # 너무 많은 티커가 있을 경우 일부만 출력
    return valid_tickers


# 단일 마켓의 데이터를 가져오고 처리하는 함수 정의
def fetch_market_data(market, index_ticker, tickers, start_date, end_date):
    print(f"Fetching data for market: {market.upper()}")

    # 인덱스 데이터 가져오기 (모든 칼럼)
    print(f"Fetching index data for {index_ticker}")
    index_data = yf.download(index_ticker, start=start_date, end=end_date)
    print(f"Index data fetched: {index_data.shape}")

    # 인덱스 데이터 인덱스 정규화 (날짜만 남기기)
    index_data.index = pd.to_datetime(index_data.index).normalize()
    print(f"Normalized index data head:\n{index_data.head()}")

    # 'Adj Close'만 선택하여 stock_data 가져오기
    print(f"Fetching stock data for {len(tickers)} tickers")
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    print(f"Stock data fetched: {stock_data.shape}")

    # 주식 데이터가 시리즈인 경우 데이터프레임으로 변환
    if isinstance(stock_data, pd.Series):
        stock_data = stock_data.to_frame()
        print("Converted Series to DataFrame for stock_data.")

    # 주식 데이터 인덱스 정규화 (날짜만 남기기)
    stock_data.index = pd.to_datetime(stock_data.index).normalize()
    print(f"Normalized stock data head:\n{stock_data.head()}")

    # 데이터 범위 확인
    print(f"Index data date range: {index_data.index.min()} to {index_data.index.max()}")
    print(f"Stock data date range: {stock_data.index.min()} to {stock_data.index.max()}")


    # 결측값 처리
    index_data = index_data.fillna(method='ffill').fillna(method='bfill')
    stock_data = stock_data.fillna(0)

    # 수익률 계산 ('Adj Close'만 사용하므로 feature는 1)
    returns = stock_data.pct_change().fillna(0)

    # 3차원 NumPy 배열 생성 [date, num_stock, feature=1]
    num_dates = stock_data.shape[0]
    num_tickers = stock_data.shape[1]
    num_features = 1  # 'Adj Close'만 사용

    print(f"Number of features: {num_features}")
    print(f"Number of tickers: {num_tickers}")

    # stocks_data_np: [date, num_stock, feature]
    stocks_data_np = stock_data.values.T.reshape((num_tickers,num_dates, num_features))

    # returns_np: [date, num_stock, feature]
    returns_np = returns.values.T

    # 'Adj Close' 데이터 2차원 데이터프레임 생성 (행: 날짜, 열: 티커)
    adj_close_df = pd.DataFrame(stock_data.values, index=stock_data.index, columns=stock_data.columns)

    # industry_classification.npy 생성: [num_tickers, num_tickers] 행렬, 각 요소는 1/num_tickers로 정규화
    industry_classification = np.full((num_tickers, num_tickers), 1 / num_tickers)

    # 마켓별 디렉토리 생성
    market_dir = f'./data/{market.upper()}'
    os.makedirs(market_dir, exist_ok=True)

    # 인덱스 데이터를 CSV로 저장 ('Adj Close'만 포함)
    index_data.to_csv(os.path.join(market_dir, f'{index_ticker}.csv'))

    # 3차원 주식 데이터를 NumPy 배열로 저장
    np.save(os.path.join(market_dir, 'stocks_data.npy'), stocks_data_np)

    # 수익률 데이터를 NumPy 배열로 저장
    np.save(os.path.join(market_dir, 'ror.npy'), returns_np)

    # 'Adj Close' 데이터 2차원 CSV로 저장
    adj_close_df.to_csv(os.path.join(market_dir, f'{market}_stocks.csv'))

    # industry_classification.npy 저장
    np.save(os.path.join(market_dir, 'industry_classification.npy'), industry_classification)

    # 저장 확인 메시지 출력
    print(f"Data for {market.upper()} saved successfully.")
    print(f"stocks_data.npy shape: {stocks_data_np.shape}")
    print(f"returns.npy shape: {returns_np.shape}")
    print(f"adj_close_data.csv shape: {adj_close_df.shape}")
    print(f"industry_classification.npy shape: {industry_classification.shape}\n")


# 마켓 구성 정의
# market_configs = {
#
#     'csi300': {
#                 'index_ticker': '000300.SS',
#                 'tickers_file': 'complete_csi300_tickers.csv',
#                 'start_date': '2014-01-01',
#                 'end_date': '2023-12-31'
#             },
# }
market_configs = {
    'kospi': {
        'index_ticker': '^KS11',
        'tickers_file': 'complete_kospi_tickers.csv',
        'start_date': '2000-01-08',
        'end_date': '2023-12-31'
    },
    'dj30': {
        'index_ticker': '^DJI',
        'tickers_file': 'complete_dj30_tickers.csv',
        'start_date': '2000-01-01',
        'end_date': '2023-12-31'
    },
    'nasdaq': {
        'index_ticker': '^IXIC',
        'tickers_file': 'complete_nasdaq_tickers.csv',
        'start_date': '2000-01-05',
        'end_date': '2023-12-31'
    },
    'csi300': {
        'index_ticker': '000300.SS',
        'tickers_file': 'complete_csi300_tickers.csv',
        'start_date': '2014-01-01',
        'end_date': '2023-12-31'
    },
}

# 각 마켓을 순회하며 데이터 가져오기
for market, config in market_configs.items():
    try:
        # CSV 파일에서 티커 로드
        tickers_df = pd.read_csv(config['tickers_file'])
        tickers = tickers_df['ticker'].astype(str).tolist()
        print(f"Loaded {len(tickers)} tickers for {market.upper()}")

        # # 티커 유효성 검증
        # tickers = validate_tickers(tickers)
        # print(f"Validated {len(tickers)} tickers for {market.upper()}")

        # 마켓 데이터 가져오고 저장
        fetch_market_data(
            market=market,
            index_ticker=config['index_ticker'],
            tickers=tickers,
            start_date=config['start_date'],
            end_date=config['end_date']
        )

        # API 속도 제한을 피하기 위해 잠시 대기
        time.sleep(1)

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}\n")
    except Exception as e:
        print(f"An error occurred while processing market {market.upper()}: {e}\n")
