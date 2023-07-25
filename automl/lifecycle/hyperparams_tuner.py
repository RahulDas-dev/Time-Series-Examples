import logging
from typing import List, Tuple

import pandas as pd
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError, MeanAbsolutePercentageError, MeanAbsoluteScaledError,
    MeanSquaredError)

from automl.lifecycle.base import Base
from automl.model_db import ModelQuery

logger = logging.getLogger(__name__)


class HyperParamsTuner(Base):
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
        self.tuned_models = []

    def get_training_test_data(self):
        test_size = len(self.fh)
        train_x = self.x.iloc[:-test_size, :]
        train_y = self.y.iloc[:-test_size]
        test_x = self.x.iloc[-test_size:, :]
        test_y = self.y.iloc[-test_size:]
        return (train_x, train_y), (test_x, test_y)

    def tune_model(self, result_df: pd.DataFrame) -> List[Tuple]:
        (train_x, train_y), _ = self.get_training_test_data()
        for idx, row in result_df.iterrows():
            logger.info(f"{idx}, Tunning Model {row['model_name']}")
            pipeline = ModelQuery.get_model_object_by_ID(self.stat, row["model_name"])
            if pipeline is None:
                continue
            grid_search = ForecastingRandomizedSearchCV(
                pipeline.forecaster,
                strategy="refit",
                scoring=self.get_scoring_metric(),
                cv=self.get_crossvalidate_spliter(len(train_y)),
                param_distributions=pipeline.hyper_parameters,
                verbose=10,
                n_jobs=-1,
                refit=True,
                error_score="raise",
                n_iter=self.random_search_iter,
            )
            grid_search.fit(train_y, X=train_x, fh=self.fh)
            # logger.info(f"Best Params {grid_search.best_params_}")
            logger.info(f"Best scores {grid_search.best_score_}")
            self.tuned_models.append(
                (
                    str(type(pipeline).identifier.name),
                    grid_search.best_forecaster_,
                    grid_search.best_score_,
                )
            )
        self.tuned_models.sort(key=lambda x: x[2])

        return self

    def get_predictions(self):
        _, (test_x, test_y) = self.get_training_test_data()
        test_y.rename("Real", inplace=True)
        y_predcitions = [test_y]
        best_model_id = self.tuned_models[0][0]
        final_model = self.tuned_models[0][1]
        for model_name, model, _ in self.tuned_models:
            logger.info(f"Predicting {model_name} .... ")
            y_pred_t = model.predict(fh=self.fh, X=test_x)
            y_pred_t.columns = [model_name]
            y_predcitions.append(y_pred_t)
        return_df = pd.concat(y_predcitions, axis=1)
        return_df["Best_model"] = return_df[best_model_id].copy()
        print(return_df)
        scoring_metrics = []
        for y_pred in y_predcitions:
            temp = {}
            mae = MeanAbsoluteError()
            temp["mae"] = mae(test_y, y_pred)
            rmse = MeanSquaredError(square_root=True)
            temp["rmse"] = rmse(test_y, y_pred)
            mape = MeanAbsolutePercentageError()
            temp["mape"] = mape(test_y, y_pred)
            # mase = MeanAbsoluteScaledError()
            # temp["mase"] = mase(test_y, y_pred)
            if y_pred.name == best_model_id:
                scoring_metrics.append({"Best_model": temp})
            scoring_metrics.append({y_pred.name: temp})
        return final_model, return_df, scoring_metrics
