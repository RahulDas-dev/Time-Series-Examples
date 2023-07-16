import logging

from automl.schuduler import Schuduler
from src.load_datasets import load_air_polution_data
from src.sanity import *

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

dataframe = (
    load_air_polution_data()
    .pipe(format_datetime, col="Date_Time")
    .pipe(set_index, col="Date_Time")
    .pipe(interpolate_column, cols="pollution")
    .pipe(resample_data, freq="H")
)

TARGET_COl = "pollution"

y = dataframe[TARGET_COl].copy(deep=True)
x = dataframe.drop(columns=[TARGET_COl]).copy(deep=True)

fh = 15
cv_split = 5

app = (
    Schuduler(y, fh)
    .set_x(x)
    .set_fh(fh)
    .set_cv_split(cv_split)
    .set_score("mae")
    .set_frequency("H")
    .extract_statistics()
    .select_model()
    .tune_models()
)
