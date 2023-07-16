from abc import ABC, abstractproperty
from enum import IntEnum, unique
from typing import Any, Dict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import (ForecastingPipeline,
                                        TransformedTargetForecaster,
                                        make_reduction)
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.compose import (OptionalPassthrough,
                                            TransformerPipeline)
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.detrend import (ConditionalDeseasonalizer,
                                                   Detrender)
from sktime.transformations.series.impute import Imputer

from src.ts_stat import SeriesStat


@unique
class ModelType(IntEnum):
    NV_MODEL = 1
    EC_MODEL = 2
    ML_MODEL = 3
    DL_MODEL = 4


@unique
class ModelID(IntEnum):
    LinearModelCCD = 1
    ElasticnetCCD = 2


class DummyForecaster(BaseForecaster):
    """Dummy Forecaster for initial  pipeline"""

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator use the exogenous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,
    }

    def _fit(self, y, X=None, fh=None):
        self._fh_len = None
        if fh is not None:
            self._fh_len = len(fh)
        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None):
        self.check_is_fitted()
        if fh is not None:
            preds = pd.Series([-99_999] * len(fh))
        elif self._fh_len is not None:
            # fh seen during fit
            preds = pd.Series([-99_999] * self._fh_len)
        else:
            raise ValueError(
                f"{type(self).__name__}: `fh` is unknown. Unable to make predictions."
            )

        return preds


class BaseModel:
    def __init__(self, stat: SeriesStat) -> None:
        super().__init__()
        self.sp = stat.primary_seasonality
        self.strictly_positive = stat.is_strickly_positive

    @classmethod
    @property
    def description(cls):
        return cls._description

    @classmethod
    @property
    def rank(cls):
        return cls._rank

    @classmethod
    @property
    def identifier(cls):
        return cls._identifier

    @classmethod
    @property
    def mtype(cls):
        return cls._mtype

    @abstractproperty
    def model(self) -> ForecastingPipeline:
        pass

    @abstractproperty
    def hyper_parameters(self) -> Dict[str, Any]:
        pass

    def find_regressor_config(self, regressor) -> Dict[str, Any]:
        regressor_args = dict()
        if hasattr(regressor, "n_jobs"):
            regressor_args["n_jobs"] = -1
        if hasattr(regressor, "random_state"):
            regressor_args["random_state"] = 80
        if hasattr(regressor, "seed"):
            regressor_args["seed"] = 80
        return regressor_args

    @property
    def forecasting_pipeline(self) -> ForecastingPipeline:
        forecaster_pipe = ForecastingPipeline(
            steps=[
                (
                    "x_transforner",
                    TransformerPipeline(
                        steps=[("imputer_x", Imputer(method="ffill", random_state=80))]
                    ),
                ),
                (
                    "scaler_x",
                    OptionalPassthrough(
                        TabularToSeriesAdaptor(MinMaxScaler()), passthrough=True
                    ),
                ),
                (
                    "forecaster",
                    TransformedTargetForecaster(
                        steps=[
                            ("imputer_y", Imputer(method="ffill", random_state=80)),
                            (
                                "deseasonalizer",
                                ConditionalDeseasonalizer(model="additive", sp=self.sp),
                            ),
                            (
                                "detrender",
                                Detrender(
                                    forecaster=PolynomialTrendForecaster(degree=1)
                                ),
                            ),
                            (
                                "reducer",
                                make_reduction(
                                    estimator=DummyForecaster(),
                                    scitype="tabular-regressor",
                                    window_length=self.sp,
                                    strategy="recursive",
                                    pooling="global",
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )
        return forecaster_pipe
