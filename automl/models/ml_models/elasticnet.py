from scipy.stats.distributions import randint, uniform

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class ElasticNet:
    def get_regressors(self):
        try:
            from sklearnex.linear_model import ElasticNet
        except ImportError:
            from sklearn.linear_model import ElasticNet
        regressor = ElasticNet()
        regressor_args = self.find_regressor_config(regressor)
        return ElasticNet(**regressor_args)


class ElasticNetModel(MLPipelineSimple, BaseModel, ElasticNet):
    _identifier: str = ModelID.Elasticnet
    _description: str = "ElasticNet "
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        if self.has_exogeneous_data:
            param_grid = {
                "scaler_x__passthrough": [True, False],
                "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
                "forecaster__reducer__estimator__alpha": uniform(0, 1),
                "forecaster__reducer__estimator__l1_ratio": uniform(0.01, 0.9999999999),
                "forecaster__reducer__estimator__max_iter": [10000],
                "forecaster__reducer__estimator__fit_intercept": [True, False],
            }
        else:
            param_grid = {
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__alpha": uniform(0, 1),
                "reducer__estimator__l1_ratio": uniform(0.01, 0.9999999999),
                "reducer__estimator__max_iter": [1000],
                "reducer__estimator__fit_intercept": [True, False],
            }
        return param_grid


class ElasticNetCCD(MLPipelineCCD, BaseModel, ElasticNet):
    _identifier: str = ModelID.ElasticnetCCD
    _description: str = "ElasticNet Conditional Deseasonalizer Detrender"
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        deseasonal_type = (
            ["additive", "multiplicative"] if self.strictly_positive else ["additive"]
        )
        if self.has_exogeneous_data:
            param_grid = {
                "scaler_x__passthrough": [True, False],
                "forecaster__deseasonalizer__model": deseasonal_type,
                # "forecaster__deseasonalizer__sp": [self.sp, 2 * self.sp],
                "forecaster__detrender__forecaster__degree": randint(1, 10),
                "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
                "forecaster__reducer__estimator__alpha": uniform(0, 1),
                "forecaster__reducer__estimator__l1_ratio": uniform(0.01, 0.9999999999),
                "forecaster__reducer__estimator__max_iter": [10000],
                # "forecaster__reducer__estimator__fit_intercept": [True, False],
            }
        else:
            param_grid = {
                "deseasonalizer__model": deseasonal_type,
                # "forecaster__deseasonalizer__sp": [self.sp, 2 * self.sp],
                "detrender__forecaster__degree": randint(1, 10),
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__alpha": uniform(0, 1),
                "reducer__estimator__l1_ratio": uniform(0.01, 0.9999999999),
                "reducer__estimator__max_iter": [1000],
                # "reducer__estimator__fit_intercept": [True, False],
            }
        return param_grid
