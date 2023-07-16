import logging

import pandas as pd
from sktime.forecasting.model_selection import ForecastingGridSearchCV

from automl.base_tunner import BaseTunner
from automl.basemodel import BaseModel
from automl.model_query import ModelQuery
from automl.models.elasticnet import ElasticNetModel
from automl.models.linearmodel import LinearModel
from automl.ts_stat import SeriesStat

logger = logging.getLogger(__name__)

model_list = [
    LinearModel,
    ElasticNetModel,
]


class ModelSelector(BaseTunner):
    def __init__(
        self,
        y: pd.Series,
        fh: int,
        x: pd.DataFrame = None,
        cv_split: int = 5,
        score: str = "mae",
    ):
        super().__init__(y, fh, x, cv_split, score)

    def select_models(self, stat: SeriesStat):
        models_list = ModelQuery.find_all_model_object(stat)
        param_grid = {
            "forecaster__reducer__estimator": [
                model.get_regressors() for model in models_list
            ]
        }
        pipeline = BaseModel(stat).forecasting_pipeline.clone()
        grid_search = ForecastingGridSearchCV(
            pipeline,
            strategy="refit",
            scoring=self.get_scoring_metric(),
            cv=self.get_crossvalidate_spliter(),
            param_grid=param_grid,
            verbose=10,
            n_jobs=-1,
            refit=True,
            error_score="raise",
            return_n_best_forecasters=1,
        )
        grid_search.fit(self.y, X=self.x, fh=self.fh)
        logger.info("Best Params", grid_search.best_params_)
        logger.info("Best scores", grid_search.best_score_)
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
