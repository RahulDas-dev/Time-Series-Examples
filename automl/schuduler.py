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
        self.settings = Settings(**settings)
        self.y, self.x, self.fh = None, None, None

    def set_y(self, y: pd.Series):
        self.y = y
        return self

    def set_x(self, x: pd.DataFrame):
        self.x = x
        return self

    def set_fh(self, fh: int):
        self.fh = fh
        return self

    def set_frequency(self, frequency: str):
        self.frequency = frequency
        return self

    def extract_statistics(self):
        logger.info("Extracting Statistics ...")
        has_exogenous = True if self.x is not None else False
        self.statistics = ExtractStats(
            self.frequency, has_exogenous
        ).extract_statistics(self.y)
        logger.info(self.statistics)
        return self

    def select_model(self):
        logger.info("Selecting Models from ...\n")
        self.model_ids, result = (
            ModelSelector(**vars(self.settings))
            .set_y(self.y)
            .set_x(self.x)
            .set_fh(self.fh)
            .select_models(self.statistics)
        )
        print(result)
        return self

    def tune_models(self):
        logger.info("Tunning Selected Models ...")
        self.tuned_model = (
            ModelTuner(**vars(self.settings))
            .set_y(self.y)
            .set_x(self.x)
            .set_fh(self.fh)
            .tune_model(self.statistics, self.model_ids)
        )
        return self

    def save_tuned_models(self):
        for model_rank, model in self.tuned_model:
            model_name = f"model_{model_rank}.pkl"
            model_path = os.path.join(self.settings.model_dir, model_name)
            logger.info(f"Saving Model ID  {model_rank} to Path {model_path}")
            joblib.dump(model, model_path)
        return self
