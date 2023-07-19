import logging

from sktime.utils import plotting

from src.load_datasets import (load_household_electricity_data,
                               load_power_consumption)
from src.sanity import create_index, resample_data
from src.ts_stat import ExtractStats, SeriesStat

logging.basicConfig(level=logging.INFO)


# dataframe = load_household_electricity_data()

dataframe = (
    load_household_electricity_data()
    .pipe(create_index, "Date_Time")
    .pipe(resample_data, freq="H")
)

print(f"dataframe shape {dataframe.shape}")

print(dataframe.info())

dataframe.head()


target_col = "Sub_metering_3"

target_data = dataframe[target_col]

# plotting.plot_series(target_data.tail(360))


ex_stat = ExtractStats(frequency="H")

stat = ex_stat.extract_statistics(target_data)
