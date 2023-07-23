import logging
import os
from typing import Any, Dict

import joblib
import pandas as pd

from automl.lifecycle.compare_model import ModelComparator
from automl.lifecycle.hyperparams_tuner import HyperParamsTuner
from automl.settings import Settings
from automl.stat.statistics import ExtractStats

logger = logging.getLogger(__name__)


class Schuduler:
    def __init__(self, settings: Dict[str, Any]):
        self._settings = Settings(**settings)
        self._exp_id = None
        self._y, self._x = None, None
        self._fh, self._frequency = None, None

    def set_exp_id(self, exp_id: str):
        self._exp_id = exp_id
        return self

    def set_y(self, y: pd.Series):
        self._y = y
        return self

    def set_x(self, x: pd.DataFrame):
        self._x = x
        return self

    def set_fh(self, fh: int):
        self._fh = fh
        return self

    def set_frequency(self, frequency: str):
        self._frequency = frequency
        return self

    def extract_statistics(self):
        logger.info("Extracting Statistics ...")
        has_exogenous = True if self._x is not None else False
        self._statistics = ExtractStats(
            self._frequency, has_exogenous
        ).extract_statistics(self._y)
        logger.info(self._statistics)
        return self

    def compare_models(self):
        logger.info("Selecting Models from ...\n")
        self._result = (
            ModelComparator(**vars(self._settings))
            .set_y(self._y)
            .set_x(self._x)
            .set_fh(self._fh)
            .set_statistics(self._statistics)
            .compare(self._settings.filter)
        )
        print(self._result)
        return self

    def tune_hyperparameters(self):
        logger.info("Tunning Selected Models ...")
        self._tuned_model, self._preictions, self._erros = (
            HyperParamsTuner(**vars(self._settings))
            .set_y(self._y)
            .set_x(self._x)
            .set_fh(self._fh)
            .set_statistics(self._statistics)
            .tune_model(self._result)
            .get_predictions()
        )
        return self

    def finalize_model(self):
        logger.info("Tunning Selected Models ...")
        print(self._preictions)
        print(self._erros)
        return self

    def save_tuned_models(self):
        logger.info("Saving Selected Tuned Models ...")
        for model_name, model in self._tuned_model:
            model_name = f"{self._exp_id}_{model_name}.pkl"
            model_path = os.path.join(self._settings.model_dir, model_name)
            logger.info(f"Saving Model ID  {model_name} to Path {model_path}")
            joblib.dump(model, model_path)
        return self

    def get_metrics(self) -> Dict:
        return self._erros

    def get_test_predictions(self) -> pd.Series:
        return self._preictions
