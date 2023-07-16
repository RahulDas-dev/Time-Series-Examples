import logging
from typing import List

import pandas as pd
from sktime.forecasting.model_selection import ForecastingGridSearchCV

from automl.base_tunner import BaseTunner
from automl.basemodel import ModelID
from automl.model_query import ModelQuery
from automl.ts_stat import SeriesStat

logger = logging.getLogger(__name__)


class ModelTuner(BaseTunner):
    def __init__(
        self,
        y: pd.Series,
        fh: int,
        x: pd.DataFrame = None,
        cv_split: int = 5,
        score: str = "mae",
    ):
        super().__init__(y, fh, x, cv_split, score)

    def tune_model(self, stat: SeriesStat, model_ids: List[ModelID]):
        model_pipelines = ModelQuery.find_all_model_object(stat, model_ids)
        tuned_model = []
        for idx, pipeline in enumerate(model_pipelines):
            logger.info(f"Tunning Model {type(pipeline).identifier}")
            grid_search = ForecastingGridSearchCV(
                pipeline.model,
                strategy="refit",
                scoring=self.get_scoring_metric(),
                cv=self.get_crossvalidate_spliter(),
                param_grid=pipeline.hyper_parameters,
                verbose=10,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            grid_search.fit(self.y, X=self.x, fh=self.fh)
            logger.info("Best Params", grid_search.best_params_)
            logger.info("Best scores", grid_search.best_score_)
            tuned_model.append(grid_search.best_forecaster_)
        return tuned_model
