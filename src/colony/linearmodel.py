try:
    from sklearnex.linear_model import LinearRegression
except ImportError:
    from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sktime.forecasting.compose import (
    ForecastingPipeline,
    make_reduction,
    TransformedTargetForecaster,
)
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.compose import TransformerPipeline
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.detrend import (
    Detrender,
    ConditionalDeseasonalizer,
)
from sktime.forecasting.trend import PolynomialTrendForecaster
from scipy.stats import randint

from src.ts_stat import SeriesStat
from src.colony.basemodel import Basemodel, ModelType


class LinearModel(Basemodel):
    def __init__(self, stat: SeriesStat):
        super().__init__(stat)
        self.__identifier: str = "Linear_CDD"
        self.__description: str = "Liner Model Conditional Deseasonalizer Detrender"
        self.__rank: int = 1
        self.__mtype: ModelType = ModelType.ML_MODEL

    @property
    def model(self) -> ForecastingPipeline:
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
                                    estimator=LinearRegression(),
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

    @property
    def hyper_parameters(self):
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__deseasonalizer__model": ["additive", "multiplicative"]
            if self.strictly_positive
            else ["additive"],
            "forecaster__deseasonalizer__sp": [self.sp, 2 * self.sp],
            "forecaster__detrender__forecaster__degree": randint(lower=1, upper=10),
            "forecaster__reducer__window_length": randint(
                lower=self.sp, upper=2 * self.sp
            ),
            "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid
