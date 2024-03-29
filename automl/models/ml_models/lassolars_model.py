from scipy.stats.distributions import loguniform, randint

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class LassoLars:
    def get_regressors(self):
        from sklearn.linear_model import LassoLars

        regressor = LassoLars()
        regressor_args = self.find_regressor_config(regressor)
        return LassoLars(**regressor_args)


class LassoLarsModel(MLPipelineSimple, BaseModel, LassoLars):
    _identifier: str = ModelID.LassoLars
    _description: str = "LassoLars Model"
    _mtype: ModelType = ModelType.LINEAR_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__alpha": loguniform(0.0000001, 1),
            "forecaster__reducer__estimator__eps": loguniform(0.00001, 0.1),
            "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid


class LassoLarsCCD(MLPipelineCCD, BaseModel, LassoLars):
    _identifier: str = ModelID.LassoLarsCCD
    _description: str = "LassoLars Model Conditional Deseasonalizer Detrender"
    _mtype: ModelType = ModelType.LINEAR_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        deseasonal_type = (
            ["additive", "multiplicative"] if self.strictly_positive else ["additive"]
        )
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__deseasonalizer__model": deseasonal_type,
            # "forecaster__deseasonalizer__sp": [self.sp, 2 * self.sp],
            "forecaster__detrender__forecaster__degree": randint(1, 10),
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__alpha": loguniform(0.0000001, 1),
            "forecaster__reducer__estimator__eps": loguniform(0.00001, 0.1),
            "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid
