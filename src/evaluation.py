from typing import List

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

__EPSILON = 1e-10


# def _error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
#    return actual - predicted


# def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
#    return np.mean(np.abs(_error(actual, predicted)))


# def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
#    return np.mean(np.square(_error(actual, predicted)))


# def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
#    return np.sqrt(mse(actual, predicted))

# def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
#    return mean_squared_error(actual, predicted, squared=False)


# def _percentage_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
#    return _error(actual, predicted) / (actual + __EPSILON)


# def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
#    return np.mean(np.abs(_percentage_error(actual, predicted)))


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
