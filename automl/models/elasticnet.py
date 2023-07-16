from scipy.stats.distributions import randint, uniform
from sktime.forecasting.compose import ForecastingPipeline

from automl.basemodel import BaseModel, ModelID, ModelType
from automl.ts_stat import SeriesStat


class ElasticNetModel(BaseModel):
    _identifier: str = ModelID.ElasticnetCCD
    _description: str = "ElasticNet Conditional Deseasonalizer Detrender"
    _rank: int = 1
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    def get_regressors(self):
        try:
            from sklearnex.linear_model import ElasticNet
        except ImportError:
            from sklearn.linear_model import ElasticNet
        regressor = ElasticNet()
        regressor_args = self.find_regressor_config(regressor)
        return ElasticNet(**regressor_args)

    @property
    def model(self) -> ForecastingPipeline:
        forecaster = self.forecasting_pipeline.clone()
        regressor = self.get_regressors()
        return forecaster.set_params(forecaster__reducer__estimator=regressor)

    @property
    def hyper_parameters(self):
        deseasonal_type = (
            ["additive", "multiplicative"] if self.strictly_positive else ["additive"]
        )
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__deseasonalizer__model": deseasonal_type,
            "forecaster__deseasonalizer__sp": [self.sp, 2 * self.sp],
            "forecaster__detrender__forecaster__degree": randint(1, 10),
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__alpha": uniform(0, 1),
            "forecaster__reducer__estimator__l1_ratio": uniform(0.01, 0.9999999999),
            "forecaster__reducer__estimator__max_iter": [1000],
            "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid
