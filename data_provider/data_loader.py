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
    def __init__(self, root_path, flag='train',valid_year=2020,test_year=2021,size=None,
                 features='M', data_path='train.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
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
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path),index_col=False)
        # df_raw = df_raw.dropna()
        df_raw = df_raw.fillna(0) #corr변수 이상 확인
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        #dependent dataset it should be fixed
        df_raw = df_raw.set_index(['date', 'tic']).sort_index()
        # df_raw = df_raw.set_index(['date','tic']).sort_index().iloc[:,1:] # for dj30

        # Split data by year
        train_data = df_raw[df_raw.index.get_level_values('date').year < self.valid_year]
        valid_data = df_raw[
            (df_raw.index.get_level_values('date').year >= self.valid_year) &
            (df_raw.index.get_level_values('date').year < self.test_year)
            ]
        test_data = df_raw[df_raw.index.get_level_values('date').year >= self.test_year]

        # Generate labels before feature normalization
        train_data = generate_labels(train_data, lookahead=self.pred_len)
        valid_data = generate_labels(valid_data, lookahead=self.pred_len)
        test_data = generate_labels(test_data, lookahead=self.pred_len)

        # Feature normalization (after label generation)

        feature_cols = train_data.columns.drop(train_data.columns[-1])  # All columns except target
        label_col = train_data.columns[-1]

        self.raw_gt = test_data[label_col].copy()
        scaler = RobustZScoreNorm(
            fit_start_time=train_data.index.get_level_values('date').min(),
            fit_end_time=train_data.index.get_level_values('date').max(),
            fields_group=feature_cols
        )
        scaler.fit(train_data)
        # Apply normalization
        train_data = scaler(train_data)
        valid_data = scaler(valid_data)
        test_data = scaler(test_data)
        # Label normalization (after label generation)
        label_scaler = CSZScoreNorm(fields_group=[label_col])

        train_data= label_scaler(train_data)
        valid_data = label_scaler(valid_data)
        test_data = label_scaler(test_data)

        if self.flag == 'train':
            data_split = train_data
        elif self.flag == 'val':
            data_split = valid_data
        elif self.flag == 'test' or self.flag =='backtest':
            data_split = test_data

            # Prepare data arrays

        self.data_x = data_split.drop(columns=[label_col])  # Input features .values
        self.gt = data_split[label_col]  # Labels .values

        # Generate time encodings
        df_stamp = data_split.reset_index()[['date']]  # Extract 'date' for time encoding
        self.df_stamp = df_stamp.drop_duplicates()  # Remove duplicates for unique dates
        if self.timeenc == 0:
            # Basic time features
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            time_features_data = df_stamp[['month', 'day', 'weekday', 'hour']].values

            # Repeat time features for each tic
            self.data_stamp = np.repeat(time_features_data, len(data_split) // len(df_stamp), axis=0)
        elif self.timeenc == 1:
            # Advanced time features
            self.data_stamp = time_features(df_stamp['date'].unique(), freq=self.freq).T #2005

            # Repeat time features for each tic
            # self.data_stamp = np.repeat(time_features_data, len(data_split) // len(df_stamp), axis=1)
        else:
            raise ValueError("Invalid value for timeenc. Must be 0 or 1.")

    def __getitem__(self, index):
        """
        Fetch data for seq_len consecutive dates, including all tics for each date.

        Parameters
        ----------
        index : int
            Index corresponding to the start date of the sequence.

        Returns
        -------
        tuple
            A tuple containing (seq_x, seq_y, seq_x_mark, seq_y_mark).
        """
        if isinstance(self.df_stamp, pd.DataFrame):
            # Extract unique dates from DataFrame
            unique_dates = self.df_stamp['date'].unique()
        elif isinstance(self.data_stamp, np.ndarray):
            # Assume ndarray contains unique dates directly
            unique_dates = np.unique(self.data_stamp)
        else:
            raise ValueError("Unsupported self.data_stamp format. Expected DataFrame or ndarray.")

        # Start and end dates for the sequence
        start_date = unique_dates[index]
        end_date = unique_dates[min(index + self.seq_len, len(unique_dates) -1)]

        # Mask to select rows for the date range
        stamp_mask = (self.df_stamp['date'] >= start_date) & (self.df_stamp['date'] < end_date)
        stamp_indices = np.where(stamp_mask)[0]

        ground_true = self.gt[self.gt.index.get_level_values('date') == end_date].values
        # Select all rows for the seq_len period
        seq_mask = (self.data_x.index.get_level_values("date") >= start_date) & (self.data_x.index.get_level_values("date") < end_date)
        seq_x_data= self.data_x[seq_mask]
        seq_x_grouped = seq_x_data.groupby(level='date')
        try:
            seq_x = np.stack([group.values for _, group in seq_x_grouped], axis=1)
        except ValueError as e:
            logger.info(f"Error stacking arrays: {e}")
        # seq_x = np.stack([group.values for _, group in seq_x_grouped], axis=1)
        seq_x_mark = np.expand_dims(self.data_stamp[stamp_indices], axis=0)
        seq_x_mark = np.tile(seq_x_mark, (seq_x.shape[0], 1, 1))

        # Labels (label_len + pred_len)
        label_start_date = unique_dates[min(index + self.seq_len - self.label_len, len(unique_dates) -1)]
        label_end_date = unique_dates[min(index + self.seq_len + self.pred_len, len(unique_dates) -1 )]

        label_stamp_mask = (self.df_stamp['date'] >= label_start_date) & (self.df_stamp['date'] < label_end_date)
        label_stamp_indices = np.where(label_stamp_mask)[0]

        label_seq_mask = (self.data_x.index.get_level_values("date") >= label_start_date) &  (self.data_x.index.get_level_values("date") < label_end_date)
        seq_y_data = self.data_x[label_seq_mask]
        seq_y_grouped = seq_y_data.groupby(level='date')
        seq_y = np.stack([group.values for _, group in seq_y_grouped], axis=1)
        seq_y_mark = np.expand_dims(self.data_stamp[label_stamp_indices], axis=0)
        seq_y_mark = np.tile(seq_y_mark, (seq_x.shape[0], 1, 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark,ground_true

    def __len__(self):

        return len(self.df_stamp) - self.seq_len - self.pred_len -1


class Dataset_Pred(Dataset):

    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
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
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_moe(Dataset):
    def __init__(self, root_path, flag='train', valid_year=2020, test_year=2021, size=None,
                 features='M', data_path='train.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
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
        self.__read_data__()

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

        # Generate labels for MOE
        train_data_moe = moe_label(train_data, pred_lens=[1, 5, 20])
        valid_data_moe = moe_label(valid_data, pred_lens=[1, 5, 20])
        test_data_moe = moe_label(test_data, pred_lens=[1, 5, 20])

        if self.flag == 'train':
            data_split = train_data
            data_split_moe = train_data_moe
        elif self.flag == 'val':
            data_split = valid_data
            data_split_moe = valid_data_moe
        elif self.flag in ['test', 'backtest']:
            data_split = test_data
            data_split_moe = test_data_moe
            self.daily_label = generate_labels(test_data, lookahead=1).iloc[:,-1]
            self.weekly_label = generate_labels(test_data, lookahead=5).iloc[:,-1]
            self.monthly_label = generate_labels(test_data, lookahead=20).iloc[:,-1]


        # Extract features and labels
        self.data_x = data_split  # Input features
        self.moe_label = data_split_moe  # Labels

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
        # Extract unique dates
        unique_dates = self.df_stamp['date'].unique()

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
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.df_stamp['date'].unique()) - self.seq_len - self.pred_len - 1