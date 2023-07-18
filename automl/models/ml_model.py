from abc import abstractproperty
from typing import Any, Dict


from sklearn.preprocessing import MinMaxScaler

from sktime.forecasting.compose import (
    ForecastingPipeline,
    TransformedTargetForecaster,
    make_reduction,
)
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.compose import OptionalPassthrough, TransformerPipeline
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.detrend import ConditionalDeseasonalizer, Detrender
from sktime.transformations.series.impute import Imputer

from automl.stat.statistics import SeriesStat
from automl.models.ml_models.customized.inputguard import ColumnsGuard
from automl.models.ml_models.customized.dummyforecaster import DummyForecaster


class MLModel:
    def __init__(self, stat: SeriesStat) -> None:
        super().__init__()
        self.sp = stat.primary_seasonality
        self.strictly_positive = stat.is_strickly_positive

    @property
    def model(self) -> ForecastingPipeline:
        forecaster = self.forecasting_pipeline.clone()
        regressor = self.get_regressors()
        return forecaster.set_params(forecaster__reducer__estimator=regressor)

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
                ("column_Gaurd", ColumnsGuard()),
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
