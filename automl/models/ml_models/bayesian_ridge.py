from scipy.stats.distributions import randint, uniform

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class BayesianRidgeModel(MLPipelineSimple, BaseModel):
    _identifier: str = ModelID.BayesianRidge
    _description: str = "Bayesian Ridge Regression"
    _mtype: ModelType = ModelType.LINEAR_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__alpha_1": uniform(0, 1),
            "forecaster__reducer__estimator__alpha_2": uniform(0, 1),
            "forecaster__reducer__estimator__lambda_1": uniform(0, 1),
            "forecaster__reducer__estimator__lambda_2": uniform(0, 1),
            "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid

    def get_regressors(self):
        from sklearn.linear_model import BayesianRidge

        regressor_args = self.find_regressor_config(BayesianRidge())
        return BayesianRidge(**regressor_args)


class BayesianRidgeCCD(MLPipelineCCD, BaseModel):
    _identifier: str = ModelID.BayesianRidgeCCD
    _description: str = "Bayesian Ridge Regression Conditional Deseasonalizer Detrender"
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
            "forecaster__detrender__forecaster__degree": randint(1, 10),
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__alpha_1": uniform(0, 1),
            "forecaster__reducer__estimator__alpha_2": uniform(0, 1),
            "forecaster__reducer__estimator__lambda_1": uniform(0, 1),
            "forecaster__reducer__estimator__lambda_2": uniform(0, 1),
        }
        return param_grid

    def get_regressors(self):
        from sklearn.linear_model import BayesianRidge

        regressor_args = self.find_regressor_config(BayesianRidge())
        return BayesianRidge(**regressor_args)
