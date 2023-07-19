import os
import urllib.request
from gzip import GzipFile
from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from src.dataset_preprocess import (process_air_polution_data,
                                    process_air_quality_data,
                                    process_electricity_data)

UCI_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases"
CURRENT_DIR: str = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR: str = os.path.join(os.path.join(CURRENT_DIR, os.pardir), "datasets")


def __down_load_zip(dataset_url: str, filename: str, seperator: str) -> pd.DataFrame:
    dataframe = None
    with urllib.request.urlopen(dataset_url) as response:
        zipfile = ZipFile(BytesIO(response.read()))
        with zipfile.open(filename, "r") as zfile:
            dataframe = pd.read_csv(zfile, sep=seperator, header=0)
    return dataframe


def __down_load_csv(dataset_url: str, seperator: str = ",") -> pd.DataFrame:
    dataframe = None
    with urllib.request.urlopen(dataset_url) as response:
        csv_content = BytesIO(response.read())
        dataframe = pd.read_csv(csv_content, sep=seperator, header=0)
    return dataframe


def __down_load_gzip(dataset_url: str, filename: str, seperator: str) -> pd.DataFrame:
    dataframe = None
    with urllib.request.urlopen(dataset_url) as response:
        with GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()
            csv_content = BytesIO(file_content)
            dataframe = pd.read_csv(csv_content, sep=seperator, header=0)
    return dataframe


def load_air_quality_data() -> pd.DataFrame:
    dataset_id = "00360"
    filename = "AirQualityUCI"
    zipfilename = f"{filename}.zip"
    csvfilename = f"{filename}.csv"
    download_Path = os.path.join(DATASET_DIR, csvfilename)

    if os.path.isfile(download_Path) is False:
        dataset_url = f"{UCI_URL}/{dataset_id}/{zipfilename}"
        # print(datset_url, csvPath)
        raw_dataframe = __down_load_zip(dataset_url, csvfilename, ";")
        dataframe = process_air_quality_data(raw_dataframe)
        dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)


def load_air_polution_data() -> pd.DataFrame:
    dataset_id = "00381"
    csvfilename = "PRSA_data_2010.1.1-2014.12.31.csv"
    download_Path = os.path.join(DATASET_DIR, csvfilename)

    if os.path.isfile(download_Path) is False:
        dataset_url = f"{UCI_URL}/{dataset_id}/{csvfilename}"
        print(f"dataset_url : {dataset_url}")
        raw_dataframe = __down_load_csv(dataset_url)
        dataframe = process_air_polution_data(raw_dataframe)
        dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)


def load_climate_change_data() -> pd.DataFrame:
    csvfilename = "jena_climate_2009_2016.csv"
    download_Path = os.path.join(DATASET_DIR, csvfilename)
    dataset_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    if os.path.isfile(download_Path) is False:
        print(f"dataset_url {dataset_url}")
        raw_dataframe = __down_load_zip(dataset_url, csvfilename, ",")
        raw_dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)


def load_power_consumption() -> pd.DataFrame:
    dataset_id = "00616"
    csvfilename = "Tetuan%20City%20power%20consumption.csv"
    download_Path = os.path.join(DATASET_DIR, csvfilename.replace("%20", "_"))
    dataset_url = f"{UCI_URL}/{dataset_id}/{csvfilename}"
    if os.path.isfile(download_Path) is False:
        raw_dataframe = __down_load_csv(dataset_url)
        raw_dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)


def load_traffic_data() -> pd.DataFrame:
    dataset_id = "00492"
    csvfilename = "Metro_Interstate_Traffic_Volume.csv"
    zipfilename = "Metro_Interstate_Traffic_Volume.csv.gz"
    download_Path = os.path.join(DATASET_DIR, csvfilename)
    dataset_url = f"{UCI_URL}/{dataset_id}/{zipfilename}"
    if os.path.isfile(download_Path) is False:
        raw_dataframe = __down_load_gzip(dataset_url, csvfilename, ",")
        raw_dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)


def load_household_electricity_data() -> pd.DataFrame:
    dataset_id = "00235"
    csvfilename = "household_power_consumption.txt"
    zipfilename = "household_power_consumption.zip"
    download_Path = os.path.join(DATASET_DIR, csvfilename.replace(".txt", ".csv"))
    dataset_url = f"{UCI_URL}/{dataset_id}/{zipfilename}"
    if os.path.isfile(download_Path) is False:
        raw_dataframe = __down_load_zip(dataset_url, csvfilename, ";")
        dataframe = process_electricity_data(raw_dataframe)
        dataframe.to_csv(download_Path, index=False)
    return pd.read_csv(download_Path)
