from typing import List, Union, Tuple
import numpy as np
import pandas as pd


datetime_default_feaures_ = [
    "hour",
    "dayofweek",
    "quarter",
    "month",
    "year",
    "dayofyear",
    "dayofmonth",
    "weekofyear",
    "dayofweek",
    "is_week_end",
    "is_week_day",
]

cyclic_default_feaures_ = ["hour", "day", "week", "month", "quarter", "year"]


def create_lag_feature(
    dataframe: pd.DataFrame, column: str, lags: Union[int, List[int]]
) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    if isinstance(dataframe_.index, pd.DatetimeIndex) is False:
        return
    dataframe_.sort_index(ascending=True)

    lags = [lags] if isinstance(lags, list) is False else lags
    for lag in lags:
        column_name = f"{column}_lag_{lag}"
        dataframe_[column_name] = dataframe_[column].shift(lag)
    return dataframe_


def create_datetime_feature(
    dataframe: pd.DataFrame, features_name: List[str]
) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    if isinstance(dataframe_.index, pd.DatetimeIndex) is False:
        return
    dataframe_.sort_index(ascending=True)

    features_name = (
        datetime_default_feaures_ if features_name is None else features_name
    )
    for feature in features_name:
        if feature == "hour":
            dataframe_["hour"] = dataframe_.index.hour
        if feature == "dayofweek":
            dataframe_["dayofweek"] = dataframe_.index.dayofweek
        if feature == "quarter":
            dataframe_["quarter"] = dataframe_.index.quarter
        if feature == "month":
            dataframe_["month"] = dataframe_.index.month
        if feature == "year":
            dataframe_["year"] = dataframe_.index.year
        if feature == "dayofyear":
            dataframe_["dayofyear"] = dataframe_.index.dayofyear
        if feature == "dayofmonth":
            dataframe_["dayofmonth"] = dataframe_.index.day
        if feature == "weekofyear":
            dataframe_["weekofyear"] = dataframe_.index.isocalendar().week
        if feature == "dayofweek":
            dataframe_["dayofweek"] = dataframe_.index.dayofweek
        if feature == "is_week_end":
            dataframe_["is_week_end"] = dataframe_.index.dayofweek > 4
            dataframe_["is_week_end"] = dataframe_["is_week_end"].apply(
                lambda x: 1 if x else 0
            )
        if feature == "is_week_day":
            dataframe_["is_week_day"] = dataframe_.index.dayofweek < 4
            dataframe_["is_week_day"] = dataframe_["is_week_day"].apply(
                lambda x: 1 if x else 0
            )
    return dataframe_


def create_cyclic_feature(
    dataframe: pd.DataFrame, features_name: List[str]
) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    if isinstance(dataframe_.index, pd.DatetimeIndex) is False:
        return
    dataframe_.sort_index(ascending=True)

    features_name = cyclic_default_feaures_ if features_name is None else features_name

    for feature in features_name:
        if feature == "hour":
            dataframe_["sin_hour"] = np.sin(dataframe_.index.hour)
            dataframe_["cos_hour"] = np.cos(dataframe_.index.hour)
        if feature == "day":
            dataframe_["sin_day"] = np.sin(dataframe_.index.day)
            dataframe_["cos_day"] = np.cos(dataframe_.index.day)
        if feature == "week":
            dataframe_["sin_week"] = np.sin(dataframe_.index.dayofweek)
            dataframe_["cos_week"] = np.cos(dataframe_.index.dayofweek)
        if feature == "month":
            dataframe_["sin_month"] = np.sin(dataframe_.index.month)
            dataframe_["cos_month"] = np.cos(dataframe_.index.month)
        if feature == "quarter":
            dataframe_["sin_quarter"] = np.sin(dataframe_.index.quarter)
            dataframe_["cos_quarter"] = np.cos(dataframe_.index.quarter)
        if feature == "year":
            dataframe_["sin_year"] = np.sin(dataframe_.index.dayofyear)
            dataframe_["cos_year"] = np.cos(dataframe_.index.dayofyear)
    return dataframe_


def create_window_feature(
    dataframe: pd.DataFrame, column: Union[str, List], window_len: int
):
    dataframe_ = dataframe.copy(deep=True)
    if isinstance(dataframe_.index, pd.DatetimeIndex) is False:
        return
    dataframe_.sort_index(ascending=True)

    column = column if isinstance(column, list) else [column]
    for col in column:
        column_name1 = f"{col}_mean_window_{window_len}"
        dataframe_[column_name1] = dataframe_[col].rolling(window=window_len).mean()
        dataframe_[column_name1].fillna(method="backfill", inplace=True)
        column_name2 = f"{col}_std_window_{window_len}"
        dataframe_[column_name2] = dataframe_[col].rolling(window=window_len).std()
        dataframe_[column_name2].fillna(method="backfill", inplace=True)
    return dataframe_


def cast_to_float(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    return dataframe_.astype(float)


def create_target_vars(
    dataframe: pd.DataFrame, target_column: str, out_len: int
) -> Tuple[pd.DataFrame, List[str]]:
    dataframe_ = dataframe.copy(deep=True)
    if isinstance(dataframe_.index, pd.DatetimeIndex) is False:
        return
    dataframe_.sort_index(ascending=True)
    target_names = [target_column]
    for i in range(1, out_len + 1):
        col_name = f"target_column_{i:02d}"
        dataframe_[col_name] = dataframe_[target_column].shift(-i)
        target_names.append(col_name)
    return dataframe_, target_names
