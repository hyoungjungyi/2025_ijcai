import os
import pandas as pd
import numpy as np
from pathlib import Path


class YfinancePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.input_path = kwargs.get('input_path', None)  # 입력 파일 경로
        self.output_path = kwargs.get('output_path', None)  # 출력 파일 경로

    def make_feature(self, df):
        """z컬럼 및 zd_* 컬럼 생성"""
        eps = 1e-12  # Zero division 방지
        periods = [5, 10, 15, 20, 25, 30, 60]

        # z 컬럼 생성
        df["zopen"] = df["open"] / (df["close"] + eps) - 1
        df["zhigh"] = df["high"] / (df["close"] + eps) - 1
        df["zlow"] = df["low"] / (df["close"] + eps) - 1
        df["zadjcp"] = df["adjclose"] / (df["close"] + eps) - 1
        df["zclose"] = df.groupby('tic')['close'].pct_change()

        # zd_* 컬럼 생성
        for period in periods:
            rolling_mean = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=period).mean())
            df[f'zd_{period}'] = rolling_mean / df['close'] - 1

        return df

    def run(self):
        """기존 데이터에서 feature 생성"""
        if not self.input_path or not os.path.exists(self.input_path):
            print(f"Input file {self.input_path} not found.")
            return

        print(f"Loading data from {self.input_path}...")
        df = pd.read_csv(self.input_path)

        # 기존 데이터에 feature 추가
        print("Generating features...")
        df = self.make_feature(df)

        # 저장 경로 확인 및 저장
        if self.output_path:
            print(f"Saving data to {self.output_path}...")
            df.to_csv(self.output_path, index=False)
            print("Data processing complete.")
        else:
            print("Output path not provided.")




config_list = [ {
        'input_path': 'raw_csi300_data.csv',
        'output_path': 'csi300_general_data.csv'

    }, {
        'input_path': 'raw_dj30_data.csv',
        'output_path': 'dj30_general_data.csv'

    },
    {
        'input_path': 'raw_kospi_data.csv',
        'output_path': 'kospi_general_data.csv'

    },
    {
        'input_path': 'raw_nasdaq_data.csv',
        'output_path': 'nasdaq_general_data.csv'

    }
]



# 데이터 처리 실행
for config in config_list:
    preprocessor = YfinancePreprocessor(**config)
    preprocessor.run()
