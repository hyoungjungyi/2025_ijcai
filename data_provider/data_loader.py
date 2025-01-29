import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
# import FinanceDataReader as fdr

from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.preprocessor import *
import warnings
import yfinance as yf
import datetime as dt
warnings.filterwarnings('ignore')




class Dataset_Custom(Dataset):
    def __init__(self, root_path='./data/dj30', flag='train',valid_year=2020,test_year=2021,size=None,
                 features='M', data_path='train.csv',
                 target='OT', scale=True, timeenc=0, freq='h',use_step_sampling=False,step_size=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val','backtest']
        self.flag = flag

        self.valid_year = valid_year
        self.test_year = test_year
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.use_step_sampling = use_step_sampling
        self.step_size = step_size if step_size is not None else self.pred_len
        self.__read_data__()
        if self.use_step_sampling:
            self.build_indexes()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw.fillna(0, inplace=True)
        # df_raw['date'] = pd.to_datetime(df_raw['date']).dt.date
        df_raw['date'] = pd.to_datetime(df_raw['date'])#.dt.date
        df_raw = df_raw.set_index(['date', 'tic']).sort_index()
        #
        # # Split data by year
        df_train = df_raw[df_raw.index.get_level_values('date').year < self.valid_year]
        df_val = df_raw[
            (df_raw.index.get_level_values('date').year >= self.valid_year) &
            (df_raw.index.get_level_values('date').year < self.test_year)
            ]
        df_test = df_raw[df_raw.index.get_level_values('date').year >= self.test_year]

        # Feature nomalization (after label generation)

        # feature_cols = train_data.columns.drop(train_data.columns[-1])  # All columns except target
        # label_col = train_data.columns[-1]
        #
        # # Define raw_gt for all splits
        # self.raw_gt = {}
        # self.raw_gt['train'] = train_data[label_col].copy()
        # self.raw_gt['val'] = valid_data[label_col].copy()
        # self.raw_gt['test'] = test_data[label_col].copy()
        # self.raw_gt['backtest'] = test_data[label_col].copy()
        # scaler = RobustZScoreNorm(
        #     fit_start_time=train_data.index.get_level_values('date').min(),
        #     fit_end_time=train_data.index.get_level_values('date').max(),
        #     fields_group=feature_cols
        # )
        # scaler.fit(train_data)
        # # Apply normalization
        # train_data = scaler(train_data)
        # valid_data = scaler(valid_data)
        # test_data = scaler(test_data)
        # # Label normalization (after label generation)
        # label_scaler = CSZScoreNorm(fields_group=[label_col])
        # train_data= label_scaler(train_data)
        # valid_data = label_scaler(valid_data)
        # test_data = label_scaler(test_data)

        if self.flag == 'train':

            df_split = df_train

        elif self.flag == 'val':

            df_split = df_val

        elif self.flag in ['test', 'backtest']:

            df_split = df_test

        elif self.flag == 'pred':

            df_split = df_test

        else:

            raise ValueError("flag must be in ['train','val','test','backtest','pred']")
        if self.use_multi_horizon:
            df_labeled = generate_labels_multiple_lookaheads(df_split, self.lookaheads)
            label_cols = [f"return_ratio_{lh}" for lh in self.lookaheads]
        else:
            df_labeled = generate_labels_single(df_split, lookahead=self.pred_len)
            label_cols = ["return_ratio"]
        feature_cols = df_labeled.columns.drop(label_cols)
        if self.scale and self.flag in ['train', 'val', 'test', 'backtest']:
            # robust scaler
            self.scaler = RobustZScoreNorm(fit_start_time=df_train.index.get_level_values('date').min(),fit_end_time=df_train.index.get_level_values('date').max(),fields_group=feature_cols)
            self.scaler.fit(df_train)
            df_labeled = self.scaler(df_labeled)
        self.df = df_labeled[feature_cols]
        self.df_label = df_labeled[label_cols]
        df_date_only = self.df.reset_index()[['date']].drop_duplicates()
        df_date_only = df_date_only.sort_values(by='date')

        if self.timeenc == 0:
                df_date_only['month'] = df_date_only['date'].dt.month

                df_date_only['day'] = df_date_only['date'].dt.day

                df_date_only['weekday'] = df_date_only['date'].dt.weekday

                df_date_only['hour'] = df_date_only['date'].dt.hour

                self.data_stamp = df_date_only.drop(columns=['date']).values  # shape: [num_dates, 4]

        else:
            time_data = time_features(df_date_only['date'].values, freq=self.freq)
            self.data_stamp = time_data.transpose(1, 0)

        self.unique_dates = df_date_only['date'].values

        self.data_x = data_split.drop(columns=[label_col])  # Input features .values
        self.gt = data_split[label_col]  # Labels .values

        # Generate time encodings
        df_stamp = data_split.reset_index()[['date']]  # Extract 'date' for time encoding
        self.df_stamp = df_stamp.drop_duplicates()  # Remove duplicates for unique dates


    def build_indexes(self):
        unique_dates = self.df_stamp['date'].unique()
        max_start = len(unique_dates) - self.seq_len - self.pred_len
        if max_start < 0:
            raise ValueError(
                f"Data is too short! Need at least seq_len+pred_len+1 days, "
                f"but only got {len(unique_dates)} unique dates."
            )

        self.indexes = list(range(0,max_start,self.step_size))

    def __getitem__(self, index):
        if self.use_step_sampling:
            index = self.indexes[index]
        # Extract unique dates
        unique_dates = self.df_stamp['date'].unique()
        if index >= len(unique_dates) - self.seq_len - self.pred_len:
            raise IndexError(f"Index {index} is out of bounds for unique_dates with length {len(unique_dates)}.")

        # Define the sequence and target date ranges
        start_date = unique_dates[index]
        end_date = unique_dates[index + self.seq_len]
        label_start_date = unique_dates[index + self.seq_len - self.label_len]
        label_end_date = unique_dates[min(index + self.seq_len  + self.pred_len, len(unique_dates) - 1)]

        # Select sequence data
        seq_mask = (self.data_x.index.get_level_values('date') >= start_date) & \
                   (self.data_x.index.get_level_values('date') < end_date)
        seq_x = self.data_x[seq_mask].groupby(level='date').apply(lambda x: x.values)
        seq_x = np.stack(seq_x, axis=1)

        # Select time encodings for sequence
        seq_time_mask = (self.df_stamp['date'] >= start_date) & (self.df_stamp['date'] < end_date)
        seq_x_mark = self.data_stamp[seq_time_mask]

        # Select label data
        label_mask = (self.data_x.index.get_level_values('date') >= label_start_date) & \
                     (self.data_x.index.get_level_values('date') < label_end_date)
        seq_y = self.data_x[label_mask].groupby(level='date').apply(lambda x: x.values)
        seq_y = np.stack(seq_y, axis=1)

        # Select time encodings for labels
        label_time_mask = (self.df_stamp['date'] >= label_start_date) & \
                          (self.df_stamp['date'] < label_end_date)
        seq_y_mark = self.data_stamp[label_time_mask]

        ground_true = self.gt[self.gt.index.get_level_values('date') == end_date].values

        return seq_x, seq_y, seq_x_mark, seq_y_mark, ground_true

    def __len__(self):
        if self.use_step_sampling:
            return len(self.indexes)
        else:
            return len(self.df_stamp) - self.seq_len - self.pred_len -1



class Dataset_moe(Dataset):
    def __init__(self, root_path, flag='train', valid_year=2020, test_year=2021, size=None,
                 features='M', data_path='train.csv',
                 target='OT', scale=True, timeenc=0, freq='h',use_step_sampling = False, step_size = None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val', 'backtest', 'moe']
        self.flag = flag

        self.valid_year = valid_year
        self.test_year = test_year
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path


        self.use_step_sampling = use_step_sampling
        self.step_size = step_size if step_size is not None else self.pred_len
        self.__read_data__()
        if self.use_step_sampling:
            self.build_indexes()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col=False)
        df_raw = df_raw.fillna(0)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.set_index(['date', 'tic']).sort_index()

        # Split data by year
        train_data = df_raw[df_raw.index.get_level_values('date').year < self.valid_year]
        valid_data = df_raw[
            (df_raw.index.get_level_values('date').year >= self.valid_year) &
            (df_raw.index.get_level_values('date').year < self.test_year)
        ]
        test_data = df_raw[df_raw.index.get_level_values('date').year >= self.test_year]

        # Generate labels before feature normalization
        train_data = generate_labels_multiple_lookaheads(train_data)
        valid_data = generate_labels_multiple_lookaheads(valid_data)
        test_data = generate_labels_multiple_lookaheads(test_data)
        # # Generate labels for MOE
        # train_data_moe = moe_label(train_data, pred_lens=[1, 5, 20])
        # valid_data_moe = moe_label(valid_data, pred_lens=[1, 5, 20])
        # test_data_moe = moe_label(test_data, pred_lens=[1, 5, 20])

        if self.flag == 'train':
            data_split = train_data
            # data_split_moe = train_data_moe
        elif self.flag == 'val':
            data_split = valid_data
            # data_split_moe = valid_data_moe
        elif self.flag in ['test', 'backtest']:
            data_split = test_data
            # data_split_moe = test_data_moe
            # self.daily_label = generate_labels(test_data, lookahead=1).iloc[:,-1]
            # self.weekly_label = generate_labels(test_data, lookahead=5).iloc[:,-1]
            # self.monthly_label = generate_labels(test_data, lookahead=20).iloc[:,-1]


        # Extract features and labels
        self.data_x = data_split  # Input features
        # self.moe_label = data_split_moe  # Labels

        # Generate time encodings
        df_stamp = data_split.reset_index()[['date']]
        self.df_stamp = df_stamp.drop_duplicates()

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            time_features_data = df_stamp[['month', 'day', 'weekday', 'hour']].values
            self.data_stamp = np.repeat(time_features_data, len(data_split) // len(df_stamp), axis=0)
        elif self.timeenc == 1:
            self.data_stamp = time_features(df_stamp['date'].unique(), freq=self.freq).T
        else:
            raise ValueError("Invalid value for timeenc. Must be 0 or 1.")

    def build_indexes(self):
        unique_dates = self.df_stamp['date'].unique()
        max_start = len(unique_dates) - self.seq_len - self.pred_len
        if max_start < 0:
            raise ValueError(
                f"Data is too short! Need at least seq_len+pred_len+1 days, "
                f"but only got {len(unique_dates)} unique dates."
            )

        self.indexes = list(range(0,max_start,self.step_size))
    def __getitem__(self, index):
        """
        Fetch data for seq_len consecutive dates, including all tics for each date,
        as well as the target sequence and its corresponding time encodings.

        Parameters
        ----------
        index : int
            Index corresponding to the start date of the sequence.

        Returns
        -------
        tuple
            A tuple containing (seq_x, seq_y, seq_x_mark, seq_y_mark, label).
        """
        if self.use_step_sampling:
            index = self.indexes[index]
        # Extract unique dates
        unique_dates = self.df_stamp['date'].unique()
        if index >= len(unique_dates):
            raise IndexError(f"Index {index} is out of bounds for unique_dates with length {len(unique_dates)}")

        # Start and end dates for the input sequence
        start_date = unique_dates[index]
        end_date = unique_dates[min(index + self.seq_len, len(unique_dates) - 1)]

        # Mask to select rows for the input sequence
        seq_mask = (self.data_x.index.get_level_values("date") >= start_date) & \
                   (self.data_x.index.get_level_values("date") < end_date)
        seq_x_data = self.data_x[seq_mask]
        seq_x_grouped = seq_x_data.groupby(level='date')
        seq_x = np.stack([group.values for _, group in seq_x_grouped], axis=1)

        # Time encodings for the input sequence
        stamp_mask = (self.df_stamp['date'] >= start_date) & (self.df_stamp['date'] < end_date)
        stamp_indices = np.where(stamp_mask)[0]
        seq_x_mark = np.expand_dims(self.data_stamp[stamp_indices], axis=0)
        seq_x_mark = np.tile(seq_x_mark, (seq_x.shape[0], 1, 1))

        # Start and end dates for the target sequence
        label_start_date = unique_dates[min(index + self.seq_len - self.label_len, len(unique_dates) - 1)]
        label_end_date = unique_dates[min(index + self.seq_len + self.pred_len, len(unique_dates) - 1)]

        # Mask to select rows for the target sequence
        label_seq_mask = (self.data_x.index.get_level_values("date") >= label_start_date) & \
                         (self.data_x.index.get_level_values("date") < label_end_date)
        seq_y_data = self.data_x[label_seq_mask]
        seq_y_grouped = seq_y_data.groupby(level='date')
        seq_y = np.stack([group.values for _, group in seq_y_grouped], axis=1)

        # Time encodings for the target sequence
        label_stamp_mask = (self.df_stamp['date'] >= label_start_date) & \
                           (self.df_stamp['date'] < label_end_date)
        label_stamp_indices = np.where(label_stamp_mask)[0]
        seq_y_mark = np.expand_dims(self.data_stamp[label_stamp_indices], axis=0)
        seq_y_mark = np.tile(seq_y_mark, (seq_x.shape[0], 1, 1))

        # MOE label (date-level)
        label = self.moe_label[end_date]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        if self.use_step_sampling:
            return len(self.indexes)
        else:
            return len(self.df_stamp) - self.seq_len - self.pred_len - 1

class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            root_path,
            data_path,
            flag='train',  # train, val, test, backtest, pred
            valid_year=2020,
            test_year=2021,
            size=None,  # (seq_len, label_len, pred_len)
            use_multi_horizon=False,
            lookaheads=[1, 5, 20],  # multi horizon list
            scale=True,
            timeenc=0,
            freq='h',
            step_size=None,
            use_step_sampling = False # step sampling 간격 (옵션)
    ):
        super().__init__()
        # size = [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.flag = flag
        self.valid_year = valid_year
        self.test_year = test_year
        self.use_multi_horizon = use_multi_horizon
        self.lookaheads = lookaheads
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path

        # step sampling
        self.step_size = step_size if step_size is not None else self.pred_len
        self.use_step_sampling = use_step_sampling

        # 데이터를 읽어서 self.df 에 저장
        # + 라벨 생성 + 스케일링 + time feature
        self.__read_data__()
        # 인덱스 빌드
        if self.use_step_sampling and self.flag != 'pred':
            self.__build_indexes__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw.fillna(0, inplace=True)
        # df_raw['date'] = pd.to_datetime(df_raw['date']).dt.date
        df_raw['date'] = pd.to_datetime(df_raw['date'])  # .dt.date
        df_raw = df_raw.set_index(['date', 'tic']).sort_index()
        #
        # # Split data by year
        df_train = df_raw[df_raw.index.get_level_values('date').year < self.valid_year]
        df_val = df_raw[
            (df_raw.index.get_level_values('date').year >= self.valid_year) &
            (df_raw.index.get_level_values('date').year < self.test_year)
            ]
        df_test = df_raw[df_raw.index.get_level_values('date').year >= self.test_year]

        if self.flag == 'train':
            df_split = df_train
        elif self.flag == 'val':
            df_split = df_val
        elif self.flag in ['test', 'backtest']:
            df_split = df_test
        elif self.flag == 'pred':

            df_split = df_test
        else:
            raise ValueError("flag must be in ['train','val','test','backtest','pred']")

            # (B) 라벨 생성
        if self.use_multi_horizon:
            # 다중 라벨
            df_labeled = generate_labels_multiple_lookaheads(df_split, self.lookaheads)
        else:
            # 단일 라벨(self.pred_len)
            df_labeled = generate_labels_single(df_split, lookahead=self.pred_len)

            # (C) 스케일링 (feature/label 분리 후 진행)
            # 일단 label 컬럼명들 정리
        if self.use_multi_horizon:
            label_cols = [f"return_ratio_{lh}" for lh in self.lookaheads]
        else:
            label_cols = ["return_ratio"]

            # 모든 열 중에서 label_cols 제외한 것 -> feature
        feature_cols = df_labeled.columns.drop(label_cols)
        # 스케일러 예시
        if self.scale and self.flag in ['train', 'val', 'test', 'backtest']:
            # robust scaler
            self.scaler = RobustZScoreNorm(
                fit_start_time=df_train.index.get_level_values('date').min(),
                fit_end_time=df_train.index.get_level_values('date').max(),
                fields_group=feature_cols
            )
            self.scaler.fit(df_train)
            df_labeled = self.scaler(df_labeled)
        self.df = df_labeled[feature_cols]
        self.df_label = df_labeled[label_cols]
        # (D) time feature
        # unique date별로 stamp 만들기
        df_date_only = self.df.reset_index()[['date']].drop_duplicates()
        df_date_only = df_date_only.sort_values(by='date')

        if self.timeenc == 0:
            # month, day, weekday, hour
            df_date_only['month'] = df_date_only['date'].dt.month
            df_date_only['day'] = df_date_only['date'].dt.day
            df_date_only['weekday'] = df_date_only['date'].dt.weekday
            df_date_only['hour'] = df_date_only['date'].dt.hour
            self.data_stamp = df_date_only.drop(columns=['date']).values  # shape: [num_dates, 4]
        else:
            # advanced time_features
            time_data = time_features(df_date_only['date'].values, freq=self.freq)
            # time_data: shape=(features, num_dates) => 전치
            self.data_stamp = time_data.transpose(1, 0)

        self.unique_dates = df_date_only['date'].values

    def __build_indexes__(self):
        """
        self.use_step_sampling=True 일 때,
        unique_dates를 기준으로 seq_len+pred_len 범위를 만들어 인덱스 리스트로 저장
        """
        max_start = len(self.unique_dates) - (self.seq_len + self.pred_len)
        if max_start < 0:
            raise ValueError("데이터가 seq_len+pred_len보다 짧습니다.")

        self.indexes = list(range(0, max_start, self.step_size))

    def __len__(self):
        # pred 모드일 경우, seq_len만큼만 뽑으면 끝나는 구조
        if self.flag == 'pred':
            # 예시) len(self.unique_dates) - self.seq_len + 1
            return len(self.unique_dates) - self.seq_len
        else:
            if self.use_step_sampling:
                return len(self.indexes)
            else:
                return len(self.unique_dates) - (self.seq_len + self.pred_len)

    def __getitem__(self, idx):
        if self.flag == 'pred':
            return self.__getitem_pred__(idx)
        else:
            return self.__getitem_train__(idx)

    def __getitem_train__(self, idx):
        """
        train/val/test/backtest 모드용 __getitem__
        """
        # step sampling이면 index를 self.indexes[idx]로 처리
        if self.use_step_sampling:
            idx = self.indexes[idx]

        start_date = self.unique_dates[idx]
        # seq 마지막 date
        seq_end_idx = idx + self.seq_len
        if seq_end_idx >= len(self.unique_dates):
            seq_end_idx = len(self.unique_dates) - 1
        seq_end_date = self.unique_dates[seq_end_idx]

        # label 구간 시작 date
        label_start_idx = seq_end_idx - self.label_len
        if label_start_idx < 0:
            label_start_idx = 0
        label_start_date = self.unique_dates[label_start_idx]

        # label 구간 끝 date
        label_end_idx = label_start_idx + self.label_len + self.pred_len
        if label_end_idx >= len(self.unique_dates):
            label_end_idx = len(self.unique_dates) - 1
        label_end_date = self.unique_dates[label_end_idx]

        # (A) seq_x
        seq_mask = (
                (self.df.index.get_level_values('date') >= start_date) &
                (self.df.index.get_level_values('date') < seq_end_date)
        )
        seq_x = self.df[seq_mask].groupby(level='date').apply(lambda x: x.values)
        seq_x = np.stack(seq_x, axis=1)  # [num_tics, seq_len, num_features]

        # time encodings for seq_x
        seq_stamp_mask = (
                (self.unique_dates >= start_date) &
                (self.unique_dates < seq_end_date)
        )
        seq_x_mark = self.data_stamp[seq_stamp_mask]


        # (B) seq_y (decoder input)
        label_mask = (
                (self.df.index.get_level_values('date') >= label_start_date) &
                (self.df.index.get_level_values('date') < label_end_date)
        )
        seq_y = self.df[label_mask].groupby(level='date').apply(lambda x: x.values)
        seq_y = np.stack(seq_y, axis=1)  # [num_tics, label_len+pred_len, num_features]

        # time encodings for seq_y
        label_stamp_mask = (
                (self.unique_dates >= label_start_date) &
                (self.unique_dates < label_end_date)
        )
        seq_y_mark = self.data_stamp[label_stamp_mask]

        # (C) ground_truth (마지막 날짜의 라벨)
        final_date = self.unique_dates[seq_end_idx]  # seq 끝 시점
        # 만약 “홀딩 끝날(=label_end_date)” 시점 라벨을 보고 싶다면,
        #    final_date를 label_end_date로 잡는 등 조정
        # 여기서는 seq 끝 시점 라벨 예시
        mask_gt = (self.df_label.index.get_level_values('date') == final_date)

        if self.use_multi_horizon:
            # 다중 라벨: return_ratio_1, return_ratio_5, return_ratio_20 ...
            label_cols = [f"return_ratio_{lh}" for lh in self.lookaheads]
            ground_true = self.df_label[mask_gt][label_cols].values  # shape: [num_tics, len(lookaheads)]
        else:
            # 단일 라벨
            ground_true = self.df_label[mask_gt]["return_ratio"].values  # shape: [num_tics]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, ground_true

    def __getitem_pred__(self, idx):
        """
        예측 모드 전용 __getitem__ 예시.
        - 단순히 seq_len 길이만큼 시계열을 가져오고,
        - pred_len 기간은 모델이 예측해야 하므로 라벨이 없다
        """
        start_idx = idx
        end_idx = start_idx + self.seq_len
        if end_idx > len(self.unique_dates):
            end_idx = len(self.unique_dates)

        start_date = self.unique_dates[start_idx]
        end_date = self.unique_dates[end_idx - 1]  # seq 마지막 날짜

        seq_mask = (
                (self.df.index.get_level_values('date') >= start_date) &
                (self.df.index.get_level_values('date') <= end_date)
        )
        seq_x = self.df[seq_mask].groupby(level='date').apply(lambda x: x.values)
        seq_x = np.stack(seq_x, axis=1)  # [num_tics, seq_len, num_features]

        # time encoding
        seq_stamp_mask = (
                (self.unique_dates >= start_date) &
                (self.unique_dates <= end_date)
        )
        seq_x_mark = self.data_stamp[seq_stamp_mask]

        # 예측 모드는 ground_truth가 없다고 가정 (혹은 test 시점에서는 있을 수 있지만)
        return seq_x, seq_x_mark