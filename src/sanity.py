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


def resample_Data(dataframe: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
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
    cols = dataframe_.columns.to_list() if cols is None else cols
    cols = cols if isinstance(cols, list) else [cols]
    for col in cols:
        dataframe_[col] = dataframe_[col].interpolate(method="backfill")
    print(f"interpolate_columns: DF Shape {dataframe_.shape}")
    return dataframe_


def fill_missing_dates(dataframe: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    ## Data  has be indexed , and freq is original data freq
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


def findSampleFreq(ts: Union[pd.Series, pd.DatetimeIndex], n: int = None):
    t_diffs = []
    n = ts.shape[0] - 1 if n is None else n
    for i in range(n):
        td = ts.iloc[i + 1] - ts.iloc[i]
        t_diffs.append(td)

    avg_diff = np.mean(t_diffs)

    # fixed with https://stackoverflow.com/a/42247228
    diff_ns = avg_diff.delta
    diff_us = diff_ns / 1000
    diff_ms = diff_us / 1000
    diff_sec = diff_ms / 1000
    diff_min = diff_sec / 60
    diff_hour = diff_min / 60
    diff_biz = (diff_hour / 24) / (7 / 5)  # assert no holidays
    diff_day = diff_hour / 24
    diff_wk = diff_day / 7
    diff_semi = diff_day / 15.21875
    diff_month = diff_day / 30.4375
    diff_qtr = diff_day / 91.3125
    diff_yr = diff_day / 365.25

    # anticipates backward timing dataframes with outer abs
    eval = {
        abs(1 - abs(k)): v
        for k, v in {
            diff_us: "U",
            diff_ms: "L",
            diff_sec: "S",
            diff_min: "T",
            diff_hour: "H",
            diff_biz: "B",
            diff_day: "D",
            diff_wk: "W",
            diff_semi: "SMS",
            diff_month: "MS",
            diff_qtr: "QS",
            diff_yr: "AS",
        }.items()
    }
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    return eval[min(eval)]
