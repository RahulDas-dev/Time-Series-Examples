import numpy as np
import pandas as pd


def process_air_quality_data(dataframe_: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe_.copy(deep=True)
    dataframe.dropna(how="all", axis=1, inplace=True)
    dataframe.dropna(inplace=True)

    dataframe["Date_Time"] = dataframe["Date"] + " " + dataframe["Time"]
    dataframe["Date_Time"] = pd.to_datetime(
        dataframe["Date_Time"], format="%d/%m/%Y %H.%M.%S"
    )
    dataframe.drop(columns=["Date", "Time"], inplace=True)
    new_columns_name = [
        "CO_true",
        "CO_sensor",
        "NMHC_true",
        "C6H6_true",
        "NMHC_sensor",
        "NOX_true",
        "NOX_sensor",
        "NO2_true",
        "NO2_sensor",
        "O3_sensor",
        "T",
        "RH",
        "AH",
        "Date_Time",
    ]
    dataframe.columns = new_columns_name
    dataframe = dataframe[new_columns_name[-1:] + new_columns_name[:-1]]
    for col in dataframe.columns:
        if dataframe[col].dtype != "O":
            continue
        dataframe[col] = pd.to_numeric(dataframe[col].str.replace(",", "."))
    return dataframe


def process_air_polution_data(dataframe_: pd.DataFrame) -> pd.DataFrame:
    dataframe_["Date_Time"] = dataframe_.apply(
        lambda x: f"{x['year']}-{x['month']}-{x['day']} {x['hour']}:00:00", axis=1
    )

    dataframe_["Date_Time"] = pd.to_datetime(
        dataframe_.pop("Date_Time"), format="%Y-%m-%d %H:%M:%S"
    )

    dataframe_ = dataframe_[
        ["Date_Time", "pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    ].copy(deep=True)

    dataframe_.rename(
        columns={
            "pm2.5": "pollution",
            "DEWP": "dewp",
            "TEMP": "temp",
            "PRES": "press",
            "cbwd": "wnd_dir",
            "Iws": "wnd_spd",
            "Is": "snow",
            "Ir": "rain",
        },
        inplace=True,
    )
    dataframe_ = dataframe_[24:]
    return dataframe_


def process_electricity_data(dataframe):
    dataframe["Date_Time"] = dataframe["Date"] + dataframe["Time"]
    dataframe["Date_Time"] = pd.to_datetime(
        dataframe["Date_Time"], format="%d/%m/%Y%H:%M:%S"
    )
    dataframe.drop(columns=["Date", "Time"], inplace=True)
    dataframe.replace("?", np.nan, inplace=True)
    for col in dataframe.columns:
        if col == "Date_Time":
            continue
        dataframe[col] = dataframe[col].astype(float)
        dataframe[col] = dataframe[col].interpolate(method="linear")

    columns_list = dataframe.columns.to_list()
    dataframe = dataframe[columns_list[-1:] + columns_list[:-1]]
    return dataframe
