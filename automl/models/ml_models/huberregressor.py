# huber_regressor.py

from scipy.stats.distributions import randint, uniform

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class HuberRegressorModel(MLPipelineSimple, BaseModel):
    _identifier: str = ModelID.HuberRegressor
    _description: str = "Huber Regressor"
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        if self.has_exogeneous_data:
            param_grid = {
                "scaler_x__passthrough": [True, False],
                "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
                "forecaster__reducer__estimator__epsilon": uniform(1.1, 2.0),
                "forecaster__reducer__estimator__max_iter": [100],
                "forecaster__reducer__estimator__alpha": uniform(0, 1),
                "forecaster__reducer__estimator__fit_intercept": [True, False],
            }
        else:
            param_grid = {
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__epsilon": uniform(1.1, 2.0),
                "reducer__estimator__max_iter": [100],
                "reducer__estimator__alpha": uniform(0, 1),
                "reducer__estimator__fit_intercept": [True, False],
            }
        return param_grid

    def get_regressors(self):
        from sklearn.linear_model import HuberRegressor

        regressor_args = self.find_regressor_config(HuberRegressor())
        return HuberRegressor(**regressor_args)


class HuberRegressorCCD(MLPipelineCCD, BaseModel):
    _identifier: str = ModelID.HuberRegressorCCD
    _description: str = "Huber Regressor Conditional Deseasonalizer Detrender"
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
                "forecaster__detrender__forecaster__degree": randint(1, 10),
                "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
                "forecaster__reducer__estimator__epsilon": uniform(1.1, 2.0),
                "forecaster__reducer__estimator__max_iter": [100],
                "forecaster__reducer__estimator__alpha": uniform(0, 1),
            }
        else:
            param_grid = {
                "deseasonalizer__model": deseasonal_type,
                "detrender__forecaster__degree": randint(1, 10),
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__epsilon": uniform(1.1, 2.0),
                "reducer__estimator__max_iter": [100],
                "reducer__estimator__alpha": uniform(0, 1),
            }
        return param_grid

    def get_regressors(self):
        from sklearn.linear_model import HuberRegressor

        regressor_args = self.find_regressor_config(HuberRegressor())
        return HuberRegressor(**regressor_args)
