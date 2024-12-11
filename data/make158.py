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
        self.start_date = kwargs.get('start_date', '2000-01-03')
        self.end_date = kwargs.get('end_date', '2023-12-29')
        self.input_path = kwargs.get('input_path', None)
        self.output_path = kwargs.get('output_path', None)
        self.indicator = kwargs.get('indicator', 'alpha158')

        # 불필요한 티커 제거
        if self.tickers:  # tickers가 None이 아닌 경우에만 필터링
            self.tickers = [ticker for ticker in self.tickers if ticker != '002030.KS']


    def fill_missing_values(self, df):
        """1999년 데이터를 이용해 결측치 채우기"""
        df['close'] = df['close'].fillna(method='ffill')
        df['open'] = df['open'].fillna(method='ffill')
        df['high'] = df['high'].fillna(method='ffill')
        df['low'] = df['low'].fillna(method='ffill')
        df['volume'] = df['volume'].fillna(method='ffill')
        df['adjclose'] = df['adjclose'].fillna(method='ffill')
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
        eps = 1e-12  # Prevent division by zero
        periods = [5, 10, 20, 30, 60]
        
        # 새로운 열을 저장할 딕셔너리
        new_features = {}

        # 기본 Feature 생성
        new_features["zopen"] = df["open"] / df["close"]
        new_features["zhigh"] = df["high"] / df["close"]
        new_features["zlow"] = df["low"] / df["close"]
        
        #vwap_ratio
        vwap = ((df['close'] * df['volume']).rolling(window=5).sum() /df['volume'].rolling(window=5).sum())
        new_features['VWAP_RATIO'] = vwap / df['close'] 


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
            # MA, STD
            new_features[f'MA{period}'] = df['close'].rolling(window=period).mean() / df['close']
            new_features[f'STD{period}'] = df['close'].rolling(window=period).std() / df['close']

            # ROC
            new_features[f'ROC{period}'] = df['close'].shift(period) / df['close']

            # RSV
            min_low = df['low'].rolling(window=period).min()
            max_high = df['high'].rolling(window=period).max()
            new_features[f'RSV{period}'] = (df['close'] - min_low) / (max_high - min_low + eps)

            # MAX, MIN
            new_features[f'MAX{period}'] = max_high / df['close']
            new_features[f'MIN{period}'] = min_low / df['close']

            # QTLU, QTLD
            new_features[f'QTLU{period}'] = df['close'].rolling(window=period).quantile(0.8) / df['close']
            new_features[f'QTLD{period}'] = df['close'].rolling(window=period).quantile(0.2) / df['close']

            # RANK
            new_features[f'RANK{period}'] = df['close'].rolling(window=period).rank(pct=True)

            # IMAX, IMIN, IMXD
            new_features[f'IMAX{period}'] = df['high'].rolling(window=period).apply(lambda x: np.argmax(x) / (period - 1), raw=True)
            new_features[f'IMIN{period}'] = df['low'].rolling(window=period).apply(lambda x: np.argmin(x) / (period - 1), raw=True)
            new_features[f'IMXD{period}'] = abs(new_features[f'IMAX{period}'] - new_features[f'IMIN{period}'])

            # BETA, RSQR, RESI
            slopes, r_squares, residuals = [], [], []
            for i in range(len(df)):
                if i < period - 1:
                    slopes.append(np.nan)
                    r_squares.append(np.nan)
                    residuals.append(np.nan)
                    continue
                y = df['close'].iloc[i - period + 1:i + 1].values
                x = np.arange(period)
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                slopes.append(slope / y[-1])  # Slope normalized by current close
                r_squares.append(r_value**2)
                residuals.append(std_err / y[-1])  # Residual normalized by current close
            new_features[f'BETA{period}'] = slopes
            new_features[f'RSQR{period}'] = r_squares
            new_features[f'RESI{period}'] = residuals

            # CORR, CORD
            log_volume = np.log(df['volume'] + 1)
            new_features[f'CORR{period}'] = df['close'].rolling(window=period).corr(log_volume)
            price_change = df['close'] / df['close'].shift(1)
            volume_change = df['volume'] / df['volume'].shift(1)
            log_volume_change = np.log(volume_change + 1)
            new_features[f'CORD{period}'] = price_change.rolling(window=period).corr(log_volume_change)

            # CNTP, CNTN, CNTD
            price_change = df['close'].diff() > 0
            new_features[f'CNTP{period}'] = price_change.rolling(window=period).mean()
            new_features[f'CNTN{period}'] = (~price_change).rolling(window=period).mean()
            new_features[f'CNTD{period}'] = new_features[f'CNTP{period}'] - new_features[f'CNTN{period}']

            # SUMP, SUMN, SUMD
            price_diff = df['close'].diff()
            positive_diff = (price_diff > 0).astype(int)
            absolute_diff = price_diff.abs()
            new_features[f'SUMP{period}'] = positive_diff.rolling(window=period).sum() / (absolute_diff.rolling(window=period).sum() + eps)
            new_features[f'SUMN{period}'] = (1 - positive_diff).rolling(window=period).sum() / (absolute_diff.rolling(window=period).sum() + eps)
            new_features[f'SUMD{period}'] = new_features[f'SUMP{period}'] - new_features[f'SUMN{period}']

            # VMA, VSTD, WVMA
            new_features[f'VMA{period}'] = df['volume'].rolling(window=period).mean() / (df['volume'] + eps)
            new_features[f'VSTD{period}'] = df['volume'].rolling(window=period).std() / (df['volume'] + eps)
            price_volatility = abs(df['close'].pct_change())
            weighted_volume = price_volatility * df['volume']
            new_features[f'WVMA{period}'] = weighted_volume.rolling(window=period).std() / (weighted_volume.rolling(window=period).mean() + eps)

            # VSUMP, VSUMN, VSUMD
            volume_diff = df['volume'].diff()
            positive_diff = (volume_diff > 0).astype(int)
            negative_diff = (volume_diff < 0).astype(int)
            absolute_diff = volume_diff.abs()
            new_features[f'VSUMP{period}'] = positive_diff.rolling(window=period).sum() / (absolute_diff.rolling(window=period).sum() + eps)
            new_features[f'VSUMN{period}'] = negative_diff.rolling(window=period).sum() / (absolute_diff.rolling(window=period).sum() + eps)
            new_features[f'VSUMD{period}'] = new_features[f'VSUMP{period}'] - new_features[f'VSUMN{period}']

        # FUT_RET
        new_features['FUT_RET5'] = df['close'].shift(-5) / df['close'].shift(-1) - 1

        # 새로운 Feature를 DataFrame에 병합
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)

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

        # 필터링: 1999년 8월 데이터부터 지표 계산
        try:
            calculation_start_date = pd.Timestamp('1999-08-01')
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
        'input_path': 'data/raw_kospi_data.csv',
        'output_path': 'data/158_kospi_data.csv',
        'start_date': '2000-01-03',
        'end_date': '2023-12-29',
        'indicator': 'alpha158'
    },
    {
        'input_path': 'data/raw_nasdaq_data.csv',
        'output_path': 'data/158_nasdaq_data.csv',
        'start_date': '2000-01-03',
        'end_date': '2023-12-29',
        'indicator': 'alpha158'
    }
]

# 데이터 처리 실행
for config in config_list:
    preprocessor = YfinancePreprocessor(**config)
    preprocessor.run()