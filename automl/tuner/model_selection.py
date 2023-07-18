import logging
from typing import List

from sktime.forecasting.model_selection import ForecastingGridSearchCV

from automl.tuner.base_tunner import BaseTunner
from automl.models.ml_model import MLModel
from automl.models.basemodel import ModelID
from automl.model_db import ModelQuery
from automl.stat.statistics import SeriesStat

logger = logging.getLogger(__name__)


class ModelSelector(BaseTunner):
    def __init__(self, model_select_count: int, cv_split: int, metric: str, **kwargs):
        super().__init__(model_select_count, cv_split, metric)
        self.y, self.x, self.fh = None, None, None

    def select_models(self, stat: SeriesStat) -> List[ModelID]:
        models_list = ModelQuery.find_all_model_object(stat)
        if len(models_list) <= self.model_select_count:
            logger.info("Skipping Model Selection ")
            return [type(model).identifier for model in models_list]

        param_grid = {
            "forecaster__reducer__estimator": [
                model.get_regressors() for model in models_list
            ]
        }
        pipeline = MLModel(stat).forecasting_pipeline.clone()

        logger.info(self.get_crossvalidate_spliter())

        grid_search = ForecastingGridSearchCV(
            pipeline,
            strategy="refit",
            scoring=self.get_scoring_metric(),
            cv=self.get_crossvalidate_spliter(),
            param_grid=param_grid,
            verbose=10,
            n_jobs=-1,
            refit=False,
            error_score="raise",
            return_n_best_forecasters=self.model_select_count,
        )
        grid_search.fit(self.y, X=self.x, fh=self.fh)
        logger.info(f"Best Params {grid_search.best_params_}")
        logger.info(f"Best scores {grid_search.best_score_}")
        regressors = grid_search.best_params_["forecaster__reducer__estimator"]

        if isinstance(regressors, list) is False:
            regressors = [regressors]

        selected_model_r_name = [
            regressor.__class__.__name__ for regressor in regressors
        ]
        model_ids = []
        for model in models_list:
            if model.get_regressors().__class__.__name__ in selected_model_r_name:
                model_ids.append(type(model).identifier)
        return model_ids
