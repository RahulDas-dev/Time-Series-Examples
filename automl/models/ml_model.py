import logging
from typing import Any, Dict

from sklearn.preprocessing import MinMaxScaler
from sktime.forecasting.compose import (ForecastingPipeline,
                                        TransformedTargetForecaster,
                                        make_reduction)
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.detrend import (ConditionalDeseasonalizer,
                                                   Detrender)
from sktime.transformations.series.impute import Imputer

from automl.models.ml_models.customized.dummyforecaster import DummyForecaster
from automl.models.ml_models.customized.inputguard import ColumnsGuard
from automl.stat.statistics import SeriesStat

logger = logging.getLogger(__name__)


class MLPipleline:
    def __init__(self, stat: SeriesStat) -> None:
        super().__init__()
        self.sp = stat.primary_seasonality
        self.strictly_positive = stat.is_strickly_positive
        self.has_exogeneous_data = stat.has_exogenous_data

    @property
    def forecaster(self) -> ForecastingPipeline:
        forecaster = self.forecasting_pipeline.clone()
        regressor = self.get_regressors()
        if self.has_exogeneous_data:
            return forecaster.set_params(forecaster__reducer__estimator=regressor)
        else:
            return forecaster.set_params(reducer__estimator=regressor)

    def find_regressor_config(self, regressor) -> Dict[str, Any]:
        regressor_args = dict()
        if hasattr(regressor, "n_jobs"):
            regressor_args["n_jobs"] = -1
        if hasattr(regressor, "random_state"):
            regressor_args["random_state"] = 80
        if hasattr(regressor, "seed"):
            regressor_args["seed"] = 80
        if hasattr(regressor, "verbose"):
            regressor["verbose"] = 0
        # logger.info(regressor_args)
        return regressor_args


class MLPipelineSimple(MLPipleline):
    def __init__(self, stat: SeriesStat) -> None:
        super().__init__(stat)

    @property
    def forecasting_pipeline(self) -> ForecastingPipeline:
        forecaster_pipe = ForecastingPipeline(
            steps=[
                ("column_Gaurd", ColumnsGuard()),
                ("imputer_x", Imputer(method="ffill", random_state=80)),
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


class MLPipelineCCD(MLPipleline):
    def __init__(self, stat: SeriesStat) -> None:
        super().__init__(stat)

    @property
    def forecasting_pipeline(self) -> ForecastingPipeline:
        forecaster_pipe = ForecastingPipeline(
            steps=[
                ("column_Gaurd", ColumnsGuard()),
                ("imputer_x", Imputer(method="ffill", random_state=80)),
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
