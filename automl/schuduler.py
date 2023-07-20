import logging
import os
from typing import Any, Dict

import joblib
import pandas as pd

from automl.settings import Settings
from automl.stat.statistics import ExtractStats
from automl.tuner.model_selection import ModelSelector
from automl.tuner.tune_model import ModelTuner

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

    def select_model(self):
        logger.info("Selecting Models from ...\n")
        self._model_ids, result = (
            ModelSelector(**vars(self._settings))
            .set_y(self._y)
            .set_x(self._x)
            .set_fh(self._fh)
            .select_models(self._statistics)
        )
        print(result)
        return self

    def tune_models(self):
        logger.info("Tunning Selected Models ...")
        self._tuned_model = (
            ModelTuner(**vars(self._settings))
            .set_y(self._y)
            .set_x(self._x)
            .set_fh(self._fh)
            .tune_model(self._statistics, self._model_ids)
        )
        return self

    def save_tuned_models(self):
        logger.info("Saving Selected Tuned Models ...")
        for model_name, model in self._tuned_model:
            model_name = f"{self._exp_id}_{model_name}.pkl"
            model_path = os.path.join(self._settings.model_dir, model_name)
            logger.info(f"Saving Model ID  {model_name} to Path {model_path}")
            joblib.dump(model, model_path)
        return self
