import yfinance as yf
import numpy as np
import pandas as pd
import os
import time


def validate_tickers(tickers):
    valid_tickers = []
    invalid_tickers = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception:
            invalid_tickers.append(ticker)
    if invalid_tickers:
        print(f"유효하지 않은 티커 발견 및 제외: {invalid_tickers[:5]} ...")
    return valid_tickers


def fetch_market_data(market, index_ticker, tickers, start_date, end_date):
    """
    인덱스 데이터는 yfinance에서 가져오고,
    주식 데이터(Adj Close)는 raw_{market}_data_filtered.csv에서 불러온 뒤,
    'stock_data'에 있는 날짜(현지 시장 기준)만 사용하여 index_data를 필터링.
    그리고 연속으로 모든 종목이 같은 값을 가지는 날(중복 행)은 최초 하루만 남기고 제거.
    """
    print(f"=== Fetching data for market: {market.upper()} ===")

    # 1) 인덱스 데이터 (yfinance)
    print(f"[Index] Fetching index data for {index_ticker}...")
    index_data = yf.download(index_ticker, start=start_date, end=end_date)
    index_data.index = pd.to_datetime(index_data.index).normalize()
    index_data.index = index_data.index.tz_localize(None)
    index_data.index.name = 'date'
    print(f"[Index] Data fetched: {index_data.shape}")

    # 2) 주식 데이터 (raw_{market}_data_filtered.csv)
    raw_csv_path = f"raw_{market}_data_filtered.csv"
    print(f"[Stocks] Loading from {raw_csv_path}")
    if not os.path.exists(raw_csv_path):
        print(f"(!) File not found: {raw_csv_path}")
        return

    raw_df = pd.read_csv(raw_csv_path)
    # 문자열 -> datetime 변환
    raw_df['date'] = pd.to_datetime(raw_df['date'])

    # 날짜가 없는 행 제거
    raw_df.dropna(subset=['date'], inplace=True)
    raw_df.sort_values(by=['date', 'tic'], inplace=True)

    # Pivot (행=날짜, 열=tic, 값=adjclose)
    stock_data = raw_df.pivot(index='date', columns='tic', values='adjclose')
    missing_mask = stock_data.isna()

    # 2) NaN이 1개 이상 존재하는지 여부
    if missing_mask.any().any():
        # 3) NaN이 True인 위치만 추출 (MultiIndex 형식)
        missing_positions = missing_mask.stack()
        missing_positions = missing_positions[missing_positions]  # True인 셀만 남김

        # 4) MultiIndex -> DataFrame으로 변환
        #    (각 행에 date, tic 컬럼이 생기고 값은 bool(True) 형태)
        missing_df = missing_positions.index.to_frame(index=False)
        # 보통 첫 번째 컬럼이 date, 두 번째 컬럼이 tic이 됨
        missing_df.columns = ['date', 'tic']

        # 5) 실제 raw_df에 존재하지 않는 (date, tic) 조합 확인
        print("=== Missing (date, tic) combinations ===")
        print(missing_df.head(30))  # 상위 30개만 출력
    else:
        print("No NaN values in pivoted data (stock_data).")
    stock_data.index = stock_data.index.normalize()
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.index.name = 'date'
    print(f"[Stocks] Loaded shape: {stock_data.shape}")

    # ----------------------------------------------------------------
    # 2-1) 날짜 교집합 (stock_data vs index_data)
    #     -> 'stock_data'를 기준(한국시간 장 열린 날)
    # ----------------------------------------------------------------
    common_dates = stock_data.index.intersection(index_data.index)
    stock_data = stock_data.loc[common_dates]
    index_data = index_data.loc[common_dates]

    print(f"[After Intersection] stock_data.shape = {stock_data.shape}, index_data.shape = {index_data.shape}")

    # ----------------------------------------------------------------
    # 2-2) '연속으로 모든 종목의 값이 동일한' 행 식별 & 제거
    # ----------------------------------------------------------------
    consecutive_dup_mask = stock_data.eq(stock_data.shift(1)).all(axis=1)
    if consecutive_dup_mask.any():
        num_dup_days = consecutive_dup_mask.sum()
        print(f"[Stocks] 연속 중복 일자 수: {num_dup_days}")
        stock_data = stock_data[~consecutive_dup_mask]

    print(f"[After Remove Duplicates] stock_data.shape = {stock_data.shape}")

    # (선택) 특정 구간(전일과 가격 동일 비율) 체크
    for col in stock_data.columns:
        daily_diff = stock_data[col].diff().fillna(0)
        zero_diff_ratio = (daily_diff == 0).mean()
        if zero_diff_ratio > 0.1:
            print(f"[Stocks] {col} : {zero_diff_ratio*100:.1f}% 날짜가 전일과 동일.")

    # 3) 수익률 계산 (중복 제거 후)
    returns = stock_data.pct_change().dropna()
    returns.index = returns.index.tz_localize(None)
    returns.index.name = 'date'

    zero_return_tickers = (returns == 0).all(axis=0)
    zero_return_tickers = zero_return_tickers[zero_return_tickers].index.tolist()
    if zero_return_tickers:
        print(f"[Stocks] 전 구간 0% 수익률 티커: {zero_return_tickers}")

    # ----------------------------------------------------------------
    # 4) 최종 날짜 인덱스 재동기화
    #    (returns는 dropna()로 인해 날짜가 하나 더 줄었을 수 있음 -> 교집합으로 맞춤)
    # ----------------------------------------------------------------
    final_dates = stock_data.index.intersection(returns.index)
    stock_data = stock_data.loc[final_dates]
    returns = returns.loc[final_dates]
    index_data = index_data.loc[final_dates]

    # ----------------------------------------------------------------
    # 5) NumPy 변환
    # ----------------------------------------------------------------
    # stocks_data: shape = [종목수, 날짜수, 1]
    stocks_data = stock_data.values.T.reshape((stock_data.shape[1], stock_data.shape[0], 1))
    # returns_np : shape = [종목수, 날짜수]
    returns_np = returns.values.T

    num_tickers = stock_data.shape[1]
    industry_classification = np.full((num_tickers, num_tickers), 1 / num_tickers)

    # ----------------------------------------------------------------
    # [추가] 최종 데이터 NaN 체크
    # ----------------------------------------------------------------
    print("\n=== Checking for NaN values in final DataFrames ===")
    for df_name, df in zip(["Index Data", "Stock Data", "Returns"], [index_data, stock_data, returns]):
        total_nan = df.isna().sum().sum()
        if total_nan > 0:
            print(f"[Warning] {df_name} contains {total_nan} NaN values.")
        else:
            print(f"{df_name} has no NaN values.")
    print("===================================================")

    # 6) 결과 저장
    market_dir = f'./data/{market.upper()}'
    os.makedirs(market_dir, exist_ok=True)

    index_csv_path = os.path.join(market_dir, f'{index_ticker}.csv')
    index_data.to_csv(index_csv_path)

    np.save(os.path.join(market_dir, 'stocks_data.npy'), stocks_data)
    np.save(os.path.join(market_dir, 'ror.npy'), returns_np)
    stock_data.to_csv(os.path.join(market_dir, f'{market.upper()}_stocks.csv'))
    np.save(os.path.join(market_dir, 'industry_classification.npy'), industry_classification)

    print(f"\nData for {market.upper()} saved successfully.")
    print(f"Index data shape: {index_data.shape}")
    print(f"Stock data shape: {stock_data.shape}")
    print(f"stocks_data.npy shape: {stocks_data.shape}")
    print(f"returns.npy shape: {returns_np.shape}")
    print(f"industry_classification.npy shape: {industry_classification.shape}\n")


# --------------------------
# 예: 마켓 구성 정의
# --------------------------
market_configs = {
    'ftse': {
        'index_ticker': '^FTSE',
        'tickers_file': 'complete_ftse_tickers.csv',
        'start_date': '2000-01-05',
        'end_date': '2023-12-31'
    },
}

for market, config in market_configs.items():
    try:
        tickers_df = pd.read_csv(config['tickers_file'])
        tickers = tickers_df['ticker'].astype(str).tolist()
        print(f"Loaded {len(tickers)} tickers for {market.upper()}")

        # (선택) 티커 유효성 검증
        # tickers = validate_tickers(tickers)
        # print(f"Validated {len(tickers)} tickers for {market.upper()}")

        fetch_market_data(
            market=market,
            index_ticker=config['index_ticker'],
            tickers=tickers,
            start_date=config['start_date'],
            end_date=config['end_date']
        )

        time.sleep(1)

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}\n")
    except Exception as e:
        print(f"An error occurred while processing market {market.upper()}: {e}\n")



# market_configs = {
#
#
# 'nasdaq': {
#     'index_ticker': '^IXIC',
#     'tickers_file': 'complete_nasdaq_tickers.csv',
#     'start_date': '2000-01-05',
#     'end_date': '2023-12-31'
# },
# 'csi300': {
#     'index_ticker': '^DJI',
#     'tickers_file': 'complete_csi300_tickers.csv',
#     'start_date': '2014-01-01',
#     'end_date': '2023-12-31'
# },
#
#
#
# 'dj30': {
#         'index_ticker': '^DJI',
#         'tickers_file': 'complete_dj30_tickers.csv',
#         'start_date': '2000-01-01',
#         'end_date': '2023-12-31'
#     },
# 'kospi': {
#         'index_ticker': '^KS11',
#         'tickers_file': 'complete_kospi_tickers.csv',
#         'start_date': '2000-01-08',
#         'end_date': '2023-12-31'
#     },
# 'ftse': {
#         'index_ticker': '^FTSE',
#         'tickers_file': 'complete_ftse_tickers.csv',
#         'start_date': '2000-01-05',
#         'end_date': '2023-12-31'
#     },
# }
