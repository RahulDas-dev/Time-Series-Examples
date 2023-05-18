from typing import List, Union
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    print(f"load_data: DF Shape {dataframe.shape}")
    return dataframe


def select_column(dataframe: pd.DataFrame, cols: Union[List[str], str]) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    if isinstance(cols, list) is False:
        cols = [cols]
    dataframe_ = dataframe_[cols]
    print(f"select_column: DF Shape {dataframe_.shape}")
    return dataframe_


def format_datetime(dataframe: pd.DataFrame, col: str, format: str) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    dataframe_[col] = pd.to_datetime(dataframe[col], format=format)
    print(f"format_datetime: DF Shape {dataframe_.shape}")
    return dataframe_


def create_index(dataframe: pd.DataFrame, col: str, format: str) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    dataframe_["Index"] = pd.to_datetime(dataframe_.pop(col), format=format)
    dataframe_.set_index(keys="Index", inplace=True)
    print(f"format_datetime: DF Shape {dataframe_.shape}")
    return dataframe_


def set_index(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    dataframe_.set_index(keys=col, inplace=True)
    print(f"set_index: DF Shape {dataframe_.shape}")
    return dataframe_


def resample_Data(dataframe: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    dataframe_ = dataframe_.resample(freq).last()
    print(f"resample_Data: DF Shape {dataframe_.shape}")
    return dataframe_


def replace_null(dataframe: pd.DataFrame, method: str = "backfill") -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    if dataframe_.isna().sum().sum() == 0:
        print(f"No Null Value Found")
        return dataframe_
    dataframe_.fillna(method=method, inplace=True)
    print(f"replace_null: DF Shape {dataframe_.shape}")
    return dataframe_


def drop_indicies(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    dataframe_.reset_index(drop=True, inplace=True)
    print(f"drop_indicies: DF Shape {dataframe_.shape}")
    return dataframe_
