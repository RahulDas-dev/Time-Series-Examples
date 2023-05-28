from io import BytesIO
from zipfile import ZipFile
import urllib.request
import os

import pandas as pd


UCI_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases"
CURRENT_DIR: str = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR: str = os.path.join(os.path.join(CURRENT_DIR, os.pardir), "datasets")


def down_load_zip(datset_url: str, filename: str, seperator: str) -> pd.DataFrame:
    response = urllib.request.urlopen(datset_url)
    zipfile = ZipFile(BytesIO(response.read()))
    with zipfile.open(filename, "r") as zfile:
        dataframe = pd.read_csv(zfile, sep=seperator, header=0)
    return dataframe


def down_load_csv(dataset_url: str, seperator: str = ",") -> pd.DataFrame:
    response = urllib.request.urlopen(dataset_url)
    csvdata = BytesIO(response.read())
    dataframe = pd.read_csv(csvdata, sep=seperator, header=0)
    return dataframe


def pre_process_air_quality_data(dataframe_: pd.DataFrame) -> pd.DataFrame:
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


def pre_process_air_polution_data(dataframe_: pd.DataFrame) -> pd.DataFrame:
    dataframe_["date"] = dataframe_.apply(
        lambda x: f"{x['year']}-{x['month']}-{x['day']} {x['hour']}:00:00", axis=1
    )

    dataframe_["date"] = pd.to_datetime(
        dataframe_.pop("date"), format="%Y-%m-%d %H:%M:%S"
    )

    dataframe_ = dataframe_[
        ["date", "pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
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


def load_air_quality_data() -> pd.DataFrame:
    dataset_id = "00360"
    filename = "AirQualityUCI"
    zipfilename = f"{filename}.zip"
    csvfilename = f"{filename}.csv"
    download_Path = os.path.join(DATASET_DIR, csvfilename)

    if os.path.isfile(download_Path) is False:
        dataset_url = f"{UCI_URL}/{dataset_id}/{zipfilename}"
        # print(datset_url, csvPath)
        raw_dataframe = down_load_zip(dataset_url, csvfilename, ";")
        dataframe = pre_process_air_quality_data(raw_dataframe)
        dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)


def load_air_polution_data() -> pd.DataFrame:
    dataset_id = "00381"
    csvfilename = "PRSA_data_2010.1.1-2014.12.31.csv"
    download_Path = os.path.join(DATASET_DIR, csvfilename)

    if os.path.isfile(download_Path) is False:
        dataset_url = f"{UCI_URL}/{dataset_id}/{csvfilename}"
        print(f"dataset_url : {dataset_url}")
        raw_dataframe = down_load_csv(dataset_url)
        dataframe = pre_process_air_polution_data(raw_dataframe)
        dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)


def load_climate_change_data() -> pd.DataFrame:
    csvfilename = "jena_climate_2009_2016.csv"
    download_Path = os.path.join(DATASET_DIR, csvfilename)
    if os.path.isfile(download_Path) is False:
        dataset_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        print(f"dataset_url {dataset_url}")
        raw_dataframe = down_load_zip(dataset_url, csvfilename, ",")
        dataframe = pre_process_air_quality_data(raw_dataframe)
        dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)
