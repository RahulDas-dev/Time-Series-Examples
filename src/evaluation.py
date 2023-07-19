from typing import List

import numpy as np
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

__EPSILON = 1e-10


__METRICS = {
    "mae": mean_absolute_error,
    "rmse": lambda r, p: mean_squared_error(r, p, squared=False),
    "mape": mean_absolute_percentage_error,
    "r2": r2_score,
}
__default_matrics = ("mae", "rmse", "mape", "r2")


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics: List[str] = None):
    results = {}
    metrics = __default_matrics if metrics is None else metrics
    for name in metrics:
        try:
            results[name] = __METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            print(f"Unable to compute metric {name}: {err}")
    return results
