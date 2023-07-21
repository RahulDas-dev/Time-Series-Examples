import logging
from typing import List

import pandas as pd
from sktime.forecasting.model_evaluation import evaluate

from automl.model_db import ModelQuery
from automl.models.basemodel import ModelID
from automl.stat.statistics import SeriesStat
from automl.tuner.base_tunner import BaseTunner

logger = logging.getLogger(__name__)


class ModelComparator(BaseTunner):
    def __init__(self, model_select_count: int, cv_split: int, metric: str, **kwargs):
        super().__init__(model_select_count, cv_split, metric)
        self.y, self.x, self.fh = None, None, None

    def compare(self, stat: SeriesStat) -> pd.DataFrame:
        models_list = ModelQuery.find_all_model_object(stat)
        if len(models_list) <= self.model_select_count:
            logger.info("Skipping Model Selection ")
            return self.build_empty_result_dir(models_list)

        logger.info(self.get_crossvalidate_spliter())
        results_list = []
        for model in models_list:
            logger.info(f"Evaluateing {type(model).identifier.name} ...")
            eval_data = evaluate(
                forecaster=model.forecaster,
                y=self.y,
                X=self.x,
                cv=self.get_crossvalidate_spliter(),
                strategy="update",
                scoring=self.get_all_scoring_matric(),
                return_data=False,
                backend="loky",
            )
            d_temp = {
                "model_id": type(model).identifier,
                "model_name": type(model).identifier.name,
                # "model_rank": type(model).rank,
                "model": model,
                "mae": eval_data["test_MeanAbsoluteError"].mean(),
                "rmse": eval_data["test_MeanSquaredError"].mean(),
                "mape": eval_data["test_MeanAbsolutePercentageError"].mean(),
                "mase": eval_data["test_MeanAbsoluteScaledError"].mean(),
                "fit_time": eval_data["fit_time"].max()
            }
            results_list.append(d_temp)
            logger.info(f"evaluate results : {d_temp[self.metric]}")
        result_df = pd.DataFrame.from_dict(results_list)
        result_df.sort_values(by=self.metric, ignore_index=True, inplace=True)
        # model_ids = final_result["model_id"].to_list()[: self.model_select_count]
        return result_df
    
    def build_empty_result_dir(self, model_list: List[ModelID]) -> pd.DataFrame:
        results_list = []
        for model in model_list:
            d_temp = {
                "model_id": type(model).identifier,
                "model_name": type(model).identifier.name,
                "model": model,
                # "model_rank": type(model).rank,
                "mae":  -1,
                "rmse": -1,
                "mape": -1,
                "mase": -1,
                "fit_time": 0
            }
            results_list.append(d_temp)
        return pd.DataFrame.from_dict(results_list)