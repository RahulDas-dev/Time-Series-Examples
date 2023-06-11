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


def format_datetime(
    dataframe: pd.DataFrame, col: str, format: str = None
) -> pd.DataFrame:
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
    dataframe_.set_index(keys=col, drop=True, inplace=True)
    dataframe_.sort_index(ascending=True)
    print(f"set_index: DF Shape {dataframe_.shape}")
    return dataframe_


def resample_data(dataframe: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    dataframe_ = dataframe_.resample(freq).mean(numeric_only=True)
    print(f"resample_Data: DF Shape {dataframe_.shape}")
    return dataframe_


def replace_null(dataframe: pd.DataFrame, method: str = "backfill") -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    if dataframe_.isna().sum().sum() == 0:
        print("No Null Value Found")
        return dataframe_
    dataframe_.fillna(method=method, inplace=True)
    print(f"replace_null: DF Shape {dataframe_.shape}")
    return dataframe_


def drop_indicies(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    dataframe_.reset_index(drop=True, inplace=True)
    print(f"drop_indicies: DF Shape {dataframe_.shape}")
    return dataframe_


def interpolate_column(dataframe: pd.DataFrame, cols: str = None) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    if dataframe.isna().sum().sum() == 0:
        return dataframe_
    cols = dataframe_.columns.to_list() if cols is None else cols
    cols = cols if isinstance(cols, list) else [cols]
    for col in cols:
        if dataframe_[col].isna().sum().sum() == 0:
            continue
        dataframe_[col] = dataframe_[col].interpolate(method="backfill")
    print(f"interpolate_columns: DF Shape {dataframe_.shape}")
    return dataframe_


def fill_missing_dates(dataframe: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    # Data has to be indexed , and freq is original data freq
    dataframe_ = dataframe.copy(deep=True)
    dataframe_ = dataframe.resample(freq).sum()
    print(f"fill_missing_dates: DF Shape {dataframe_.shape}")
    return dataframe_


def cast_datetime_column(dataframe: pd.DataFrame):
    from pandas.errors import ParserError

    dataframe_ = dataframe.copy(deep=True)
    for c in dataframe_.columns[dataframe_.dtypes == "object"]:
        try:
            dataframe_[c] = pd.to_datetime(
                dataframe_[c], infer_datetime_format=True
            )  # fixing times
        except (ParserError, ValueError):
            try:
                dataframe_[c] = (
                    dataframe_[c].str.replace(",", "").astype("float64")
                )  # fixing numbers
            except (ParserError, ValueError):
                pass
    return dataframe_


def put_target_columns_to_end(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    other_cols = set(dataframe_.columns.tolist()) - set([col])
    rearranged_col = list(other_cols) + [col]
    dataframe_ = dataframe_[rearranged_col]
    return dataframe_


def remove_space_from_columns_name(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    new_columns = {col: str(col).strip() for col in dataframe_.columns.tolist()}
    dataframe_.rename(columns=new_columns, inplace=True)
    return dataframe_


def rename_column(dataframe: pd.DataFrame, col: str, new_column: str) -> pd.DataFrame:
    dataframe_ = dataframe.copy(deep=True)
    dataframe_.rename(columns={col: new_column}, inplace=True)
    return dataframe_
