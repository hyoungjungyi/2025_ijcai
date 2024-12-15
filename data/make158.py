import os
import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np
from scipy.stats import linregress

class YfinancePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tickers = kwargs.get('tickers', [])  # None 대신 빈 리스트로 기본값 설정
        self.start_date = kwargs.get('start_date', '2000-01-01')
        self.end_date = kwargs.get('end_date', '2023-12-31')
        self.input_path = kwargs.get('input_path', None)
        self.output_path = kwargs.get('output_path', None)
        self.indicator = kwargs.get('indicator', 'alpha158')

        # 불필요한 티커 제거
        if self.tickers:  # tickers가 None이 아닌 경우에만 필터링
            self.tickers = [ticker for ticker in self.tickers if ticker != '002030.KS']

    def fill_missing_values(self, df):
        """결측치 및 0값 처리"""
        df['close'] = df['close'].replace(0, np.nan).ffill()
        df['open'] = df['open'].replace(0, np.nan).ffill()
        df['high'] = df['high'].replace(0, np.nan).ffill()
        df['low'] = df['low'].replace(0, np.nan).ffill()
        df['volume'] = df['volume'].replace(0, np.nan).fillna(1)
        df['adjclose'] = df['adjclose'].replace(0, np.nan).ffill()
        return df
    
    def make_alphafeature(self, df):
        """기본 feature 생성"""
        df["zopen"] = df["open"] / df["close"] 
        df["zhigh"] = df["high"] / df["close"] 
        df["zlow"] = df["low"] / df["close"] 
        df["zadjcp"] = df["adjclose"] / df["close"] 
        return df

    def make_158feature(self, df):
        """
        종가, 거래량, 고가, 저가 등을 기반으로 다양한 기술적 지표를 생성하는 함수.
        """
        # 1. 데이터 정렬
        df = df.sort_values(by=['tic', 'date']).reset_index(drop=True)
        eps = 1e-12  # Prevent division by zero

        periods = [5, 10, 20, 30, 60]
        
        # 새로운 열을 저장할 딕셔너리
        new_features = {}

        # 기본 Feature 생성
        new_features["zopen"] = df["open"] / df["close"]
        new_features["zhigh"] = df["high"] / df["close"]
        new_features["zlow"] = df["low"] / df["close"]

        # VWAP_RATIO

        vwap = (df['close'] * df['volume']).groupby(df['tic']).rolling(window=5).sum() / df['volume'].groupby(df['tic']).rolling(window=5).sum()
        df['VWAP_RATIO'] = vwap.reset_index(level=0, drop=True) / df['close']


        # 추가 Feature 생성
        new_features['Daily_Return'] = (df['close'] - df['open']) / df['open']
        new_features['KLEN'] = (df['high'] - df['low']) / df['open']
        new_features['KMID2'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + eps)
        new_features['KUP'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        new_features['KUP2'] = (df['high'] - np.maximum(df['open'], df['close'])) / ((df['high'] - df['low']) + eps)
        new_features['KLOW'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        new_features['KLOW2'] = (np.minimum(df['open'], df['close']) - df['low']) / ((df['high'] - df['low']) + eps)
        new_features['KSFT'] = (2 * df['close'] - df['high'] - df['low']) / df['open']
        new_features['KSFT2'] = (2 * df['close'] - df['high'] - df['low']) / ((df['high'] - df['low']) + eps)

        # 롤링 윈도우 기반 Feature 생성
        for period in periods:
            grouped = df.groupby('tic')

            # Rolling objects
            close_rolling = grouped['close'].rolling(window=period)
            low_rolling = grouped['low'].rolling(window=period)
            high_rolling = grouped['high'].rolling(window=period)
            volume_rolling = grouped['volume'].rolling(window=period)

            # Precomputed rolling statistics
            rolling_mean_close = close_rolling.mean().reset_index(level=0, drop=True)
            rolling_std_close = close_rolling.std().reset_index(level=0, drop=True)
            rolling_min_low = low_rolling.min().reset_index(level=0, drop=True)
            rolling_max_high = high_rolling.max().reset_index(level=0, drop=True)

            # MA, STD
            new_features[f'MA{period}'] = rolling_mean_close / df['close']
            new_features[f'STD{period}'] = rolling_std_close / df['close']

            # ROC (Rate of Change)
            new_features[f'ROC{period}'] = grouped['close'].transform(lambda x: x / x.shift(period) - 1)

            # RSV
            new_features[f'RSV{period}'] = (df['close'] - rolling_min_low) / (rolling_max_high - rolling_min_low + eps)

            # MAX, MIN
            new_features[f'MAX{period}'] = rolling_max_high / df['close']
            new_features[f'MIN{period}'] = rolling_min_low / df['close']

            # QTLU, QTLD
            new_features[f'QTLU{period}'] = close_rolling.quantile(0.8).reset_index(level=0, drop=True) / df['close']
            new_features[f'QTLD{period}'] = close_rolling.quantile(0.2).reset_index(level=0, drop=True) / df['close']

            # RANK
            new_features[f'RANK{period}'] = close_rolling.rank(pct=True).reset_index(level=0, drop=True)

            # IMAX, IMIN, IMXD
            new_features[f'IMAX{period}'] = high_rolling.apply(lambda x: np.argmax(x) / (period - 1),
                                                               raw=True).reset_index(level=0, drop=True)
            new_features[f'IMIN{period}'] = low_rolling.apply(lambda x: np.argmin(x) / (period - 1),
                                                              raw=True).reset_index(level=0, drop=True)
            new_features[f'IMXD{period}'] = abs(new_features[f'IMAX{period}'] - new_features[f'IMIN{period}'])

            # BETA, RSQR, RESI
            def linreg_slope(y):
                if len(y) < 2:  # Ensure at least two data points
                    return np.nan
                slope, _, _, _, _ = linregress(range(len(y)), y)
                return slope

            def linreg_r_squared(y):
                if len(y) < 2:
                    return np.nan
                _, _, r_value, _, _ = linregress(range(len(y)), y)
                return r_value ** 2

            def linreg_stderr(y):
                if len(y) < 2:
                    return np.nan
                _, _, _, _, stderr = linregress(range(len(y)), y)
                return stderr

            # Apply rolling calculations
            slope_series = close_rolling.apply(linreg_slope, raw=False).reset_index(level=0, drop=True)
            r_squared_series = close_rolling.apply(linreg_r_squared, raw=False).reset_index(level=0, drop=True)
            stderr_series = close_rolling.apply(linreg_stderr, raw=False).reset_index(level=0, drop=True)

            # Assign features
            new_features[f'BETA{period}'] = slope_series / df['close']
            new_features[f'RSQR{period}'] = r_squared_series
            new_features[f'RESI{period}'] = stderr_series / df['close']

            # CORR, CORD
            new_features[f'CORR{period}'] = grouped.apply(lambda g: g['close'].rolling(window=period).corr(np.log(g['volume'] + 1))).reset_index(level=0, drop=True)
            new_features[f'CORD{period}'] = grouped.apply(lambda g: (g['close'] / g['close'].shift(1)).rolling(window=period).corr(np.log((g['volume'] / g['volume'].shift(1)) + 1))).reset_index(level=0, drop=True)
            # new_features[f'CORR{period}'] = grouped['close'].apply(lambda x: x.rolling(window=period).corr(np.log(grouped['volume'].transform(lambda v: v.loc[x.index] + 1)))).reset_index(level=0, drop=True)
            # new_features[f'CORD{period}'] = grouped['close'].pct_change().apply(lambda x: x.rolling(window=period).corr(np.log(grouped['volume'].pct_change().transform(lambda v: v.loc[x.index] + 1)))).reset_index(level=0, drop=True)




            # CNTP, CNTN, CNTD
            price_diff = grouped['close'].diff()  # Ticker별 차이를 계산
            positive_diff = (price_diff > 0).astype(int)
            negative_diff = (price_diff < 0).astype(int)
            absolute_diff = price_diff.abs()

            cnt_positive = positive_diff.groupby(df['tic']).rolling(window=period).mean().reset_index(level=0,
                                                                                                      drop=True)
            cnt_negative = negative_diff.groupby(df['tic']).rolling(window=period).mean().reset_index(level=0,
                                                                                                      drop=True)

            new_features[f'CNTP{period}'] = cnt_positive
            new_features[f'CNTN{period}'] = cnt_negative
            new_features[f'CNTD{period}'] = cnt_positive - cnt_negative

            # SUMP, SUMN, SUMD
            sump = positive_diff.groupby(df['tic']).rolling(window=period).sum().reset_index(level=0, drop=True)
            sumn = negative_diff.groupby(df['tic']).rolling(window=period).sum().reset_index(level=0, drop=True)
            sum_abs = absolute_diff.groupby(df['tic']).rolling(window=period).sum().reset_index(level=0, drop=True)

            new_features[f'SUMP{period}'] = sump / (sum_abs + eps)
            new_features[f'SUMN{period}'] = sumn / (sum_abs + eps)
            new_features[f'SUMD{period}'] = new_features[f'SUMP{period}'] - new_features[f'SUMN{period}']

            # VMA, VSTD, WVMA
            rolling_mean_volume = volume_rolling.mean().reset_index(level=0, drop=True)
            rolling_std_volume = volume_rolling.std().reset_index(level=0, drop=True)

            new_features[f'VMA{period}'] = rolling_mean_volume / (df['volume'] + eps)
            new_features[f'VSTD{period}'] = rolling_std_volume / (df['volume'] + eps)

            price_volatility = grouped['close'].pct_change().abs()
            weighted_volume = price_volatility * df['volume']
            weighted_std = weighted_volume.groupby(df['tic']).rolling(window=period).std().reset_index(level=0,
                                                                                                       drop=True)
            weighted_mean = weighted_volume.groupby(df['tic']).rolling(window=period).mean().reset_index(level=0,
                                                                                                         drop=True)

            new_features[f'WVMA{period}'] = weighted_std / (weighted_mean + eps)

            # VSUMP, VSUMN, VSUMD
            volume_diff =grouped['volume'].diff()
            pos_diff_vol = (volume_diff > 0).astype(int)
            neg_diff_vol = (volume_diff < 0).astype(int)
            abs_diff_vol = volume_diff.abs()

            vsump = pos_diff_vol.groupby(df['tic']).rolling(window=period).sum().reset_index(level=0, drop=True)
            vsumn = neg_diff_vol.groupby(df['tic']).rolling(window=period).sum().reset_index(level=0, drop=True)
            vsum_abs = abs_diff_vol.groupby(df['tic']).rolling(window=period).sum().reset_index(level=0, drop=True)

            new_features[f'VSUMP{period}'] = vsump / (vsum_abs + eps)
            new_features[f'VSUMN{period}'] = vsumn / (vsum_abs + eps)
            new_features[f'VSUMD{period}'] = new_features[f'VSUMP{period}'] - new_features[f'VSUMN{period}']



        # 새로운 Feature를 DataFrame에 병합
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
        df = df.sort_values(by=['date', 'tic']).reset_index(drop=True)

        return df

    def sort_columns_by_variable(self, df):
        """칼럼 순서를 변수별로 정렬"""
        base_columns = [col for col in df.columns if not any(c.isdigit() for c in col)]  # 날짜 등 기본 컬럼
        rolling_columns = [col for col in df.columns if any(c.isdigit() for c in col)]  # 윈도우 컬럼
        
        # 정렬 기준: 변수 이름 + 숫자
        rolling_columns_sorted = sorted(
            rolling_columns,
            key=lambda x: (''.join(filter(str.isalpha, x)), int(''.join(filter(str.isdigit, x))))
        )
        return df[base_columns + rolling_columns_sorted]


    def run(self):
        """데이터 처리"""
        if not self.input_path:
            print("Input path not provided.")
            return

        print(f"Loading data from {self.input_path}...")
        try:
            df = pd.read_csv(self.input_path)
        except Exception as e:
            print(f"Failed to load data: {e}")
            return

        print(f"Initial data shape: {df.shape}")

        # 날짜 변환 및 타입 확인
        try:
            print("Sample raw date values before conversion:")
            print(df['date'].head())  # 변환 전 샘플 출력

            # 'date' 열을 datetime으로 변환
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            print("Sample date values after conversion:")
            print(df['date'].head())  # 변환 후 샘플 출력
            print(f"Date column type after conversion: {df['date'].dtypes}")
        except Exception as e:
            print(f"Error during date conversion: {e}")
            return

        # NaN 값 제거
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"Number of invalid dates: {invalid_dates}")
            print("Dropping invalid date rows...")
            df = df.dropna(subset=['date'])
        print(f"After dropping invalid dates: {df.shape}")

        # 타임존 제거
        try:
            if pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = df['date'].dt.tz_localize(None)  # 타임존 제거
                print("Timezone successfully removed.")
            else:
                raise ValueError("Date column is not in datetime64 format after conversion.")
        except Exception as e:
            print(f"Error during timezone removal: {e}")
            return


        try:
            calculation_start_date = pd.Timestamp('2000-01-01')
            df = df[df['date'] >= calculation_start_date]
            print(f"Data filtered for calculation starting from {calculation_start_date}: {df.shape}")
        except Exception as e:
            print(f"Error during calculation date filtering: {e}")
            return

        if df.empty:
            print("No data available after calculation date filtering.")
            return

        # 결측치 채우기
        print("Filling missing values...")
        df = self.fill_missing_values(df)
        print(f"After filling missing values: {df.isna().sum().sum()} missing values remaining.")

        # Feature 생성
        print("Generating features...")
        df = self.make_158feature(df)

        if df.empty:
            print("DataFrame is empty after feature generation. No data to save.")
            return

        # 필터링: 2000년 1월 1일 데이터부터 저장
        try:
            storage_start_date = pd.Timestamp('2000-01-01')
            df = df[df['date'] >= storage_start_date]
            print(f"Data filtered for storage starting from {storage_start_date}: {df.shape}")
        except Exception as e:
            print(f"Error during storage date filtering: {e}")
            return

        if df.empty:
            print("No data available after storage date filtering.")
            return

        # 컬럼 정렬
        df = self.sort_columns_by_variable(df)

        # 디버깅: 최종 컬럼 확인
        print(f"Final column count: {len(df.columns)}")
        print(f"Final column names: {df.columns.tolist()}")

        # 데이터 저장
        if self.output_path:
            try:
                print(f"Saving data to {self.output_path}...")
                df.to_csv(self.output_path, index=False)
                print(f"Data saved to {self.output_path}")
                print(f"Shape of saved data: {df.shape}")
            except Exception as e:
                print(f"Failed to save data: {e}")
        else:
            print("Output path not provided.")


# 데이터 처리 설정


config_list = [
    {
        'input_path': 'raw_kospi_data.csv',
        'output_path': 'kospi_alpha158_data.csv',
        'start_date': '2000-01-01',
        'end_date': '2023-12-31',
        'indicator': 'alpha158'
    },
    {
        'input_path': 'raw_nasdaq_data.csv',
        'output_path': 'nasdaq_alpha158_data.csv',
        'start_date': '2000-01-01',
        'end_date': '2023-12-31',
        'indicator': 'alpha158'
    }
]

# 데이터 처리 실행
for config in config_list:
    preprocessor = YfinancePreprocessor(**config)
    preprocessor.run()