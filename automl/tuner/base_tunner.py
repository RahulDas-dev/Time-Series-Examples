import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanSquaredError,
)


class BaseTunner:
    def __init__(self, model_select_count: int, cv_split: int, metric: str):
        self.model_select_count = model_select_count
        self.cv_split = cv_split
        self.metric = metric
        self.y, self.x, self.fh = None, None, None

    def set_y(self, y: pd.Series):
        self.y = y
        return self

    def set_x(self, x: pd.DataFrame):
        self.x = x
        return self

    def set_fh(self, fh: int):
        self.fh = np.arange(1, fh + 1)
        return self

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
        if self.metric == "mae":
            return MeanAbsoluteError()
        elif self.metric == "rmse":
            return MeanSquaredError(square_root=True)
        elif self.metric == "mse":
            return MeanSquaredError(square_root=False)
        elif self.metric == "mape":
            return MeanAbsolutePercentageError(symmetric=False)
        elif self.metric == "smape":
            return MeanAbsolutePercentageError(symmetric=True)
        elif self.metric == "mase":
            return MeanAbsoluteScaledError()
        elif self.metric == "all":
            return [MeanAbsoluteError(),
                    MeanSquaredError(square_root=True),
                    MeanSquaredError(square_root=False),
                    MeanAbsolutePercentageError(symmetric=False),
                    MeanAbsolutePercentageError(symmetric=True),
                    MeanAbsoluteScaledError()]
        else:
            return MeanAbsoluteError()
