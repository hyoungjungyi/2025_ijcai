import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# 티커 리스트 불러오기 (필요에 따라 수정)
csi300_tickers = [str(ticker) for ticker in pd.read_csv('complete_csi300_tickers.csv')['ticker'].tolist()]
dj30_tickers   = [str(ticker) for ticker in pd.read_csv('complete_dj30_tickers.csv')['ticker'].tolist()]
kospi_tickers  = [str(ticker) for ticker in pd.read_csv('complete_kospi_tickers.csv')['ticker'].tolist()]
#nasdaq_tickers = [str(ticker) for ticker in pd.read_csv('complete_nasdaq_tickers.csv')['ticker'].tolist()]
ftse_tickers = [str(ticker) for ticker in pd.read_csv('complete_ftse_tickers.csv')['ticker'].tolist()]
def format_tickers(ticker_list):
    return ", ".join(ticker_list)

print("CSI 300 Tickers:", format_tickers(csi300_tickers))
print("DJ30 Tickers:",   format_tickers(dj30_tickers))
print("KOSPI Tickers:",  format_tickers(kospi_tickers))
#print("NASDAQ Tickers:", format_tickers(nasdaq_tickers))
print("FTSE Tickers:", format_tickers(ftse_tickers))

def fetch_and_save_ticker_data(ticker_list,
                               output_csv,
                               start_date=None,
                               end_date=None,
                               threshold=0.5,
                               output_filtered_tickers_csv=None,
                               output_removed_tickers_csv=None):
    """
    1) Ticker별 yfinance 데이터를 다운로드 후,
       - (a) 0% 수익률 비중(threshold) 초과하는 '종목' 제거
       - (b) 'adjclose'를 기준으로 직전 날짜와 모든 티커 값이 완전히 동일한 날짜(연속 중복) 제거
         -> 제거는 최종 데이터 전체( open/close/high/low/volume 포함 )에서도 해당 날짜 row를 제거
    2) 최종 결과를 CSV로 저장, 제거된/유지된 티커 목록도 별도 CSV로 저장 가능

    Args:
      ticker_list (list): List of stock tickers
      output_csv (str): Path to save the resulting CSV file
      start_date (str): start date (YYYY-MM-DD)
      end_date (str): end date (YYYY-MM-DD)
      threshold (float): Zero-return 비율 기준
      output_filtered_tickers_csv (str): 필터링 후 남은 티커 저장 CSV
      output_removed_tickers_csv (str): 제거된 티커 + 그 비중 저장 CSV
    """
    print(f"\n=== fetch_and_save_ticker_data (threshold={threshold}) ===")
    all_data = []

    # ----------------------
    # 1) 티커별 데이터 다운로드
    # ----------------------
    for ticker in ticker_list:
        try:
            stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False)
            # print(ticker,stock_data.index[-1])
            if stock_data.empty:
                print(f"No data found for {ticker}, skipping...")
                continue

            stock_data.reset_index(inplace=True)

            # 'Adj Close' 보정
            if 'Adj Close' not in stock_data.columns:
                if 'Dividends' in stock_data.columns and not stock_data['Dividends'].isnull().all():
                    stock_data['Adj Close'] = stock_data['Close'] - stock_data['Dividends']
                    print('stock adj check (Dividends) -> ', ticker)
                else:
                    stock_data['Adj Close'] = stock_data['Close']
                    print('stock adj check -> ', ticker)

            # 컬럼명 통일
            stock_data = stock_data[['Date', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume']].copy()
            stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'adjclose', 'volume']
            stock_data['tic'] = ticker

            # datetime 변환 및 TZ 제거
            stock_data['date'] = pd.to_datetime(stock_data['date'], utc=True).dt.normalize()

            all_data.append(stock_data)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if not all_data:
        print("No data to save (all_data is empty).")
        return

    # ----------------------
    # 2) Long Format 통합
    # ----------------------
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df[['date', 'tic', 'open', 'close', 'high', 'low', 'adjclose', 'volume']]
    final_df.sort_values(by=['date', 'tic'], inplace=True)



    # *** 모든 티커가 동일한 날짜 인덱스를 갖도록 재인덱싱 ***
    # full_dates = pd.date_range(start=final_df['date'].min(), end=final_df['date'].max())
    # corrected_data = []
    # for ticker in final_df['tic'].unique():
    #     tic_df = final_df[final_df['tic'] == ticker].copy()
    #     tic_df.set_index('date', inplace=True)
    #     # full_dates로 reindex하고, 결측치는 전일 값(forward fill)으로 채우기
    #     tic_df = tic_df.reindex(full_dates)
    #     tic_df = tic_df.fillna(method='ffill')
    #     tic_df['tic'] = ticker
    #     tic_df.reset_index(inplace=True)
    #     tic_df.rename(columns={'index': 'date'}, inplace=True)
    #     corrected_data.append(tic_df)
    # final_df = pd.concat(corrected_data, ignore_index=True)

    # ----------------------
    # 3) 0% 수익률 과다 종목 제거
    # ----------------------
    final_df['daily_return'] = final_df.groupby('tic')['adjclose'].pct_change().fillna(0)
    final_df['is_zero_return'] = (final_df['daily_return'] == 0)
    zero_ratio_by_tic = final_df.groupby('tic')['is_zero_return'].mean()

    removed_info = zero_ratio_by_tic[zero_ratio_by_tic > threshold].sort_values(ascending=False)
    removed_tickers = removed_info.index.tolist()

    kept_info = zero_ratio_by_tic[zero_ratio_by_tic <= threshold].sort_values(ascending=False)
    kept_tickers = kept_info.index.tolist()

    print("\n--- Zero Return Ratio Distribution (All Tickers) ---")
    print(zero_ratio_by_tic.describe())  # 통계치
    print(f"\nThreshold: {threshold * 100:.1f}%")
    print(f"Removed {len(removed_tickers)} tickers:")
    for i, (tic, ratio) in enumerate(removed_info.items()):
        print(f"  {i+1}. {tic} -> {ratio * 100:.2f}% zero-return")
    print(f"\nKept {len(kept_tickers)} tickers.\n")

    # 제거 대상 티커 drop
    filtered_df = final_df[~final_df['tic'].isin(removed_tickers)].copy()
    for date, group in filtered_df.groupby('date'):
        unique_tickers = group['tic'].unique()
        print(f"날짜: {date}, 티커 개수: {len(unique_tickers)}")
    shape_before_consecutive = filtered_df.shape

    # 임시로 drop
    filtered_df.drop(['daily_return', 'is_zero_return'], axis=1, inplace=True)

    # ----------------------
    # 4) "연속 중복 날짜" 제거 (adjclose 기준)
    #    4-1) Wide Format (pivot) 변환
    # ----------------------
    pivot_df = filtered_df.pivot(index='date', columns='tic', values='adjclose')
    pivot_df.sort_index(inplace=True)

    print(f"[Before consecutive removal] pivot_df shape: {pivot_df.shape}")

    # ----------------------
    # 4-2) 연속 중복 검사
    #   - 현재 date행이 바로 이전 date행과 완전히 동일한지(모든 칼럼=티커 값이 동일한지)
    # ----------------------
    consecutive_dup_mask = pivot_df.eq(pivot_df.shift(1)).all(axis=1)
    num_dup_days = consecutive_dup_mask.sum()
    print(f"Found {num_dup_days} consecutive duplicate days. Removing them...")

    # 유지할 날짜
    remaining_dates = pivot_df.index[~consecutive_dup_mask]  # False인 date만 유지

    # pivot_df.shape에서 제거 후
    pivot_df_filtered = pivot_df.loc[remaining_dates]
    print(f"[After consecutive removal] pivot_df shape: {pivot_df_filtered.shape}")

    # ----------------------
    # 4-3) Long Format에서 해당 날짜만 남김
    # ----------------------
    # => open/close/high/low/volume 모두 유지
    filtered_df = filtered_df[filtered_df['date'].isin(remaining_dates)]

    shape_after_consecutive = filtered_df.shape
    removed_rows = shape_before_consecutive[0] - shape_after_consecutive[0]
    print(f"Removed {removed_rows} rows by consecutive duplicate removal.")
    print(f"Shape before = {shape_before_consecutive}, Shape after = {shape_after_consecutive}")

    # ----------------------
    # 5) 최종 CSV 저장
    # ----------------------
    if filtered_df.empty:
        print("All data removed; final_df is empty.")
    else:
        # 여기서 open/close/high/low/volume, adjclose 모두 포함
        filtered_df.to_csv(output_csv, index=False)
        print(f"Filtered data (with O/C/H/L/V) saved to {output_csv}")

    # ----------------------
    # 6) 제거된/유지된 티커 목록 CSV 저장
    # ----------------------
    if output_filtered_tickers_csv:
        pd.DataFrame({'kept_tickers': kept_tickers}).to_csv(output_filtered_tickers_csv, index=False)
        print(f"Kept Tickers saved to {output_filtered_tickers_csv}")

    if output_removed_tickers_csv:
        removed_df = pd.DataFrame({'removed_tickers': removed_info.index,
                                   'zero_return_ratio': removed_info.values})
        removed_df.to_csv(output_removed_tickers_csv, index=False)
        print(f"Removed Tickers saved to {output_removed_tickers_csv}")


# ==============================
# 실제 함수 호출 (시장별 처리)
# ==============================
today=datetime.today().date()
if __name__ == "__main__":
    threshold_val = 0.1

    fetch_and_save_ticker_data(
         ticker_list=csi300_tickers,
         output_csv="raw_csi300_data_filtered.csv",
         start_date='2014-01-01',
         end_date=today,
         threshold=threshold_val,
         output_filtered_tickers_csv="csi300_kept_tickers.csv",
         output_removed_tickers_csv="csi300_removed_tickers.csv"
     )
    
    fetch_and_save_ticker_data(
         ticker_list=dj30_tickers,
         output_csv="raw_dj30_data_filtered.csv",
         start_date='2000-01-01',
         end_date=today,
         threshold=threshold_val,
         output_filtered_tickers_csv="dj30_kept_tickers.csv",
         output_removed_tickers_csv="dj30_removed_tickers.csv"
     )

    fetch_and_save_ticker_data(
         ticker_list=kospi_tickers,
         output_csv="raw_kospi_data_filtered.csv",
         start_date='2000-01-07',
         end_date=today,
         threshold=threshold_val,
         output_filtered_tickers_csv="kospi_kept_tickers.csv",
        output_removed_tickers_csv="kospi_removed_tickers.csv"
     )

    #fetch_and_save_ticker_data(
         #ticker_list=nasdaq_tickers,
         #output_csv="raw_nasdaq_data_filtered.csv",
         #start_date='2000-01-05',
         #end_date=today,
         #threshold=threshold_val,
         #output_filtered_tickers_csv="nasdaq_kept_tickers.csv",
         #output_removed_tickers_csv="nasdaq_removed_tickers.csv"
     #)
    fetch_and_save_ticker_data(
        ticker_list=ftse_tickers,
        output_csv="raw_ftse_data_filtered.csv",
        start_date='2000-01-05',
        end_date=today,
        threshold=threshold_val,
        output_filtered_tickers_csv="ftse_kept_tickers.csv",
        output_removed_tickers_csv="ftse_removed_tickers.csv"
    )

    # 최종 확인
    print("\n=== Checking Results ===")
    for csv_name in [
        "raw_csi300_data_filtered.csv",
        "raw_dj30_data_filtered.csv",
        "raw_kospi_data_filtered.csv",
        "raw_nasdaq_data_filtered.csv",
        "raw_ftse_data_filtered.csv"
    ]:
        if os.path.exists(csv_name):
            df_check = pd.read_csv(csv_name)
            print(f"\nData {csv_name} -> shape = {df_check.shape}")
            print(df_check.head(5))
        else:
            print(f"{csv_name} not found.")
