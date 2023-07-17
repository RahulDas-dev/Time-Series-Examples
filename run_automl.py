import logging
import os
import warnings

from automl.schuduler import Schuduler
from src.load_datasets import load_air_polution_data, load_traffic_data
from src.sanity import (
    format_datetime,
    interpolate_column,
    resample_data,
    select_column,
    set_index,
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def get_air_polution_data():
    dataframe = (
        load_air_polution_data()
        .pipe(format_datetime, col="Date_Time")
        .pipe(set_index, col="Date_Time")
        .pipe(interpolate_column, cols="pollution")
        .pipe(resample_data, freq="H")
    )
    null_count = dataframe.isna().sum().sum()
    logger.info(f"NULL Count {null_count}")
    y = dataframe.pop("pollution")
    return y, dataframe


def get_traffiC_data():
    dataframe = (
        load_traffic_data()
        .pipe(
            select_column,
            cols=[
                "date_time",
                "temp",
                "rain_1h",
                "snow_1h",
                "clouds_all",
                "traffic_volume",
            ],
        )
        .pipe(format_datetime, col="date_time")
        .pipe(set_index, col="date_time")
        .pipe(resample_data, freq="H")
        .pipe(interpolate_column)
    )
    null_count = dataframe.isna().sum().sum()
    print(dataframe.head())
    logger.info(f"NULL Count {null_count}")
    y = dataframe.pop("traffic_volume")
    return y, dataframe


if __name__ == "__main__":
    # y, x = get_air_polution_data()
    
    y, x = get_traffiC_data()
    fh = 15
    settings = {
        "model_dir": os.path.abspath("./results"),
        "model_select_count": 1,
        "cv_split": 5,
        "metric": "mase",
    }
    app = (
        Schuduler(settings)
        .set_y(y)
        .set_x(x)
        .set_fh(fh)
        .set_frequency("H")
        .extract_statistics()
        .select_model()
        .tune_models()
        .save_tuned_models()
    )
