import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError, MeanAbsolutePercentageError, MeanAbsoluteScaledError,
    MeanSquaredError)


class BaseTunner:
    def __init__(
        self,
        y: pd.Series,
        fh: int,
        x: pd.DataFrame = None,
        cv_split: int = 5,
        score: str = "mae",
    ):
        self.y = y
        self.x = x
        self.fh = np.arange(1, fh + 1)
        self.cv_split = cv_split
        self.score = score

    def get_crossvalidate_spliter(self):
        y_size = len(self.y)
        step_length = len(self.fh)
        fh_max_length = max(self.fh)
        intital_window_size = y_size - (
            (self.cv_split - 1) * step_length + 1 * fh_max_length
        )
        cv = ExpandingWindowSplitter(
            self.fh, initial_window=intital_window_size, step_length=step_length
        )
        return cv

    def get_scoring_metric(self):
        if self.score == "mae":
            return MeanAbsoluteError()
        elif self.score == "rmse":
            return MeanSquaredError(square_root=True)
        elif self.score == "mse":
            return MeanSquaredError(square_root=False)
        elif self.score == "mape":
            return MeanAbsolutePercentageError(symmetric=False)
        elif self.score == "smape":
            return MeanAbsolutePercentageError(symmetric=True)
        elif self.score == "mase":
            return MeanAbsoluteScaledError()
        else:
            return MeanAbsoluteError()
