import pandas as pd
import numpy as np
import abc
from typing import Union, Text, Optional
from scipy.stats import zscore as standard_zscore

EPS = 1e-12


def moe_label(df, pred_lens):
    """
    Generate MOE labels by calculating cumulative returns for multiple prediction intervals.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with index ['date', 'tic'] and a 'close' column.
    pred_lens : list of int
        List of prediction intervals (e.g., [1, 5, 20]).

    Returns
    -------
    moe_label : pd.Series
        A Series of MOE labels as integers [0, 1, 2] corresponding to pred_lens.
    cumulative_returns_df : pd.DataFrame
        Cumulative returns for each prediction interval, indexed by date.
    """
    df = df.sort_index()

    # 가장 긴 pred_len을 기준으로 청산 횟수 계산
    max_len = max(pred_lens)
    cumulative_returns = {}

    for pred_len in pred_lens:
        num_intervals = max_len // pred_len  # 최대 기간에서 몇 번 청산할지 결정
        portfolio_value = pd.Series(1, index=df.index)

        for i in range(num_intervals):  # 청산 횟수만큼 반복
            shift_start = df.groupby("tic")["close"].shift(-1 - i * pred_len)  # 시작 가격
            shift_end = df.groupby("tic")["close"].shift(-1 - (i + 1) * pred_len)  # 청산 가격
            interval_return = (shift_end - shift_start) / shift_start
            interval_return = interval_return.fillna(0)

            # 재투자 논리 추가
            daily_gains = (1 + interval_return).groupby(level='date').mean()  # 종목별 평균
            portfolio_value.loc[daily_gains.index] *= daily_gains

        # 청산 기간 전체 누적 수익률 저장
        cumulative_returns[pred_len] = portfolio_value.groupby(level='date').sum()

    # 누적 수익률을 DataFrame으로 합치기
    cumulative_returns_df = pd.DataFrame(cumulative_returns)

    # 최적의 투자 기간을 moe_label로 설정
    moe_label = cumulative_returns_df.idxmax(axis=1)  # 최적의 pred_len 선택
    moe_label = moe_label.map({val: idx for idx, val in enumerate(pred_lens)})  # [1, 5, 20] -> [0, 1, 2]

    # 유효하지 않은 행 제거
    moe_label = moe_label.dropna().astype(int)  # 정수형 레이블로 변환
    return moe_label



def generate_labels(df, lookahead=5):
    """
    Generate labels for stock price forecasting based on the given formula.
    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with index ['date', 'tic'] and a 'close' column.
    lookahead : int
        Prediction interval (number of days ahead to calculate return ratio).
    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'return_ratio' column (not normalized).
    """
    # Ensure the DataFrame is sorted by MultiIndex ('date', 'tic')
    df = df.sort_index()

    # Calculate the return ratio using (c[τ+d] - c[τ+1]) / c[τ+1]
    df["close_shift_1"] = df.groupby("tic")["close"].shift(-1)  # c[τ+1]
    df["close_shift_d"] = df.groupby("tic")["close"].shift(-1 - lookahead)  # c[τ+d]
    df["return_ratio"] = (df["close_shift_d"] - df["close_shift_1"]) / df["close_shift_1"]

    # Drop intermediate calculation columns
    df = df.drop(columns=["close_shift_1", "close_shift_d"])

    # Drop rows where return ratio cannot be calculated
    return df.dropna(subset=["return_ratio"])


def get_level_index(df: pd.DataFrame, level: Union[str, int]) -> int:
    """
    Helper function to get the numeric index of a level in a MultiIndex.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    level : Union[int, str]
        The level name (str) or index (int).

    Returns
    -------
    int
        The numeric index of the level.
    """
    if isinstance(level, int):
        return level
    elif isinstance(level, str):
        if level in df.index.names:
            return df.index.names.index(level)
        raise ValueError(f"Level '{level}' not found in MultiIndex names: {df.index.names}")
    else:
        raise TypeError(f"Level must be an integer or string, got {type(level)}")


def fetch_df_by_index(
        df: pd.DataFrame,
        selector: Union[pd.Timestamp, slice, str, list, pd.Index],
        level: Union[str, int],
        fetch_orig: bool = True,
) -> pd.DataFrame:
    """
    Fetch data from a DataFrame with the given selector and level.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to fetch data from.
    selector : Union[pd.Timestamp, slice, str, list, pd.Index]
        The index selector to use for filtering.
    level : Union[int, str]
        The level in the MultiIndex to filter on.
    fetch_orig : bool, optional
        If True, return the original DataFrame if no filtering is performed
        (default is True).

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.

    Raises
    ------
    ValueError
        If the level is invalid or not found in the MultiIndex.
    TypeError
        If the selector or level has an unsupported type.

    Examples
    --------
    >>> arrays = [
    ...     ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    ...     ["AAPL", "GOOG", "AAPL", "GOOG"],
    ... ]
    >>> index = pd.MultiIndex.from_arrays(arrays, names=["date", "ticker"])
    >>> data = pd.DataFrame({"price": [150, 2800, 155, 2825]}, index=index)
    >>> fetch_df_by_index(data, selector="2024-01-01", level="date")
                       price
    date       ticker
    2024-01-01 AAPL     150
               GOOG    2800
    """
    # If no specific level is provided or selector is already a MultiIndex
    if level is None or isinstance(selector, pd.MultiIndex):
        return df.loc(axis=0)[selector]

    # Validate and get the numeric level index
    level_idx = get_level_index(df, level)

    # Create index slice for filtering
    if level_idx == 0:
        idx_slc = (selector, slice(None, None))
    else:
        idx_slc = (slice(None, None), selector)

    # Handle fetch_orig logic
    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[pd.IndexSlice[idx_slc]]
        return df
    else:
        return df.loc[pd.IndexSlice[idx_slc]]



def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    """
    get a group of columns from multi-index columns DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        with multi of columns.
    group : str
        the name of the feature group, i.e. the first level value of the group index.
    """
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]


class RobustZScoreNorm:
    """Robust Z-Score Normalization.

    This normalization method uses robust statistics:
        - mean(x) = median(x)
        - std(x) = MAD(x) * 1.4826

    Parameters
    ----------
    fit_start_time : str
        The start time of the fitting period.
    fit_end_time : str
        The end time of the fitting period.
    fields_group : list[str], optional
        List of column names to normalize. Default is None, meaning all columns.
    clip_outlier : bool, optional
        Whether to clip the outliers to the range [-3, 3]. Default is True.

    Reference
    ---------
    - https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group=None, clip_outlier=True):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        self.mean_train = None
        self.std_train = None

    def fit(self, df: pd.DataFrame):
        # Validate input dates
        if self.fit_start_time >= self.fit_end_time:
            raise ValueError("fit_start_time must be earlier than fit_end_time.")

        # Select data within the fitting range
        df = df.loc[self.fit_start_time:self.fit_end_time]

        # Validate columns
        self.cols = self.fields_group if self.fields_group is not None else df.columns.tolist()
        if not set(self.cols).issubset(df.columns):
            raise ValueError(f"fields_group contains invalid columns: {self.fields_group}")

        # Calculate robust statistics
        X = df[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train = mad * 1.4826
        self.std_train = np.maximum(self.std_train, 1e-8)  # Prevent division by zero

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_train is None or self.std_train is None:
            raise ValueError("The fit method must be called before using this processor.")

        # Apply normalization
        X = df[self.cols]
        X -= self.mean_train
        X /= self.std_train

        # Clip outliers if required
        if self.clip_outlier:
            X = np.clip(X, -3, 3)

        # Update the DataFrame
        df[self.cols] = X
        return df


class CSZScoreNorm:
    """Cross-Sectional Z-Score Normalization with Optional Outlier Removal.

    This method normalizes data for each time group (e.g., date) using either
    standard Z-Score or robust Z-Score. Optionally, it removes the top and bottom
    5% of values from the entire dataset before normalization.

    Parameters
    ----------
    fields_group : list[str]
        List of column groups to normalize.
    method : str, optional
        Normalization method, "zscore" (default) or "robust".
    remove_outliers : bool, optional
        Whether to remove the top and bottom 5% of values across the entire dataset
        before normalization. Default is False.
    """

    def __init__(self, fields_group=None, method="zscore", remove_outliers=False):
        if fields_group is None or not isinstance(fields_group, list):
            raise ValueError("fields_group must be a non-empty list of column names.")
        self.fields_group = fields_group
        self.remove_outliers = remove_outliers
        self.zscore_func = standard_zscore


    def remove_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Remove top and bottom 5% of values for a column."""
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to the given DataFrame."""
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("The DataFrame must have a MultiIndex with 'date' and 'tic'.")

        # Ensure MultiIndex contains 'date' and 'tic'
        if 'date' not in df.index.names or 'tic' not in df.index.names:
            raise ValueError("The MultiIndex must contain 'date' and 'tic' levels.")

        df = df.copy()  # Avoid modifying the original DataFrame

        for col in self.fields_group:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

            # Step 1: Optionally remove outliers across all rows for this column
            if self.remove_outliers:
                df = self.remove_outliers(df, col)

            # Step 2: Apply Z-Score normalization by date (cross-sectional)
            df[col] = (
                df.groupby(level="date", group_keys=False)[col]
                    .transform(self.zscore_func)
            )
        return df


class CSRankNorm:
    """Cross-Sectional Rank Normalization for MultiIndex DataFrames.

    This method normalizes data for each time group (e.g., date) by ranking the
    data within each group, converting it to a uniform distribution, and then normalizing
    to approximate a standard normal distribution.

    Parameters
    ----------
    fields_group : Union[str, list[str]]
        Column name or list of column names to normalize.

    Example
    -------
    >>> processor = CSRankNorm(fields_group="feature1")
    >>> normalized_df = processor(df)
    """

    def __init__(self, fields_group):
        # Allow single column name or list of column names
        if isinstance(fields_group, str):
            fields_group = [fields_group]
        if not isinstance(fields_group, list) or len(fields_group) == 0:
            raise ValueError("fields_group must be a non-empty list or a single column name as a string.")
        self.fields_group = fields_group

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rank normalization to the given MultiIndex DataFrame."""
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("The DataFrame must have a MultiIndex with 'date' and 'tic'.")

        # Ensure MultiIndex contains 'date' and 'tic'
        if 'date' not in df.index.names or 'tic' not in df.index.names:
            raise ValueError("The MultiIndex must contain 'date' and 'tic' levels.")

        df = df.copy()  # Avoid modifying the original DataFrame

        for col in self.fields_group:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

            # Rank normalization by date level
            t = df.groupby(level="date", group_keys=False)[col].rank(pct=True)
            t -= 0.5  # Center around 0
            t *= 3.46  # Scale to unit standard deviation
            df[col] = t

        return df

