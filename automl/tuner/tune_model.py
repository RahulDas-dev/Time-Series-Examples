import logging
from typing import List, Tuple

from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV

from automl.model_db import ModelQuery
from automl.models.basemodel import ModelID
from automl.stat.statistics import SeriesStat
from automl.tuner.base_tunner import BaseTunner

logger = logging.getLogger(__name__)


class ModelTuner(BaseTunner):
    def __init__(
        self,
        model_select_count: int,
        cv_split: int,
        metric: str,
        random_search_iter: int,
        **kawrgs,
    ):
        super().__init__(model_select_count, cv_split, metric)
        self.random_search_iter = random_search_iter

    def tune_model(self, stat: SeriesStat, model_ids: List[ModelID]) -> List[Tuple]:
        model_pipelines = ModelQuery.find_all_model_object(stat, model_ids)
        tuned_model = []
        for idx, pipeline in enumerate(model_pipelines):
            logger.info(f"Tunning Model {type(pipeline).identifier}")

            grid_search = ForecastingRandomizedSearchCV(
                pipeline.forecaster,
                strategy="refit",
                scoring=self.get_scoring_metric(),
                cv=self.get_crossvalidate_spliter(),
                param_distributions=pipeline.hyper_parameters,
                verbose=10,
                n_jobs=-1,
                refit=True,
                error_score="raise",
                n_iter=self.random_search_iter,
            )
            grid_search.fit(self.y, X=self.x, fh=self.fh)
            logger.info(f"Best Params {grid_search.best_params_}")
            logger.info(f"Best scores {grid_search.best_score_}")
            tuned_model.append((type(pipeline).rank, grid_search.best_forecaster_))
        return tuned_model
