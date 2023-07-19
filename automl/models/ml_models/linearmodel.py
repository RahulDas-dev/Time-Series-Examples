from scipy.stats.distributions import randint

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class Linear:
    def get_regressors(self):
        try:
            from sklearnex.linear_model import LinearRegression
        except ImportError:
            from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor_args = self.find_regressor_config(regressor)
        return LinearRegression(**regressor_args)


class LinearModel(MLPipelineSimple, BaseModel, Linear):
    _identifier: str = ModelID.Linear
    _description: str = "Liner Model "
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        if self.has_exogeneous_data:
            param_grid = {
                "scaler_x__passthrough": [True, False],
                "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
                "forecaster__reducer__estimator__fit_intercept": [True, False],
            }
        else:
            param_grid = {
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__fit_intercept": [True, False],
            }
        return param_grid


class LinearModelCCD(MLPipelineCCD, BaseModel, Linear):
    _identifier: str = ModelID.LinearCCD
    _description: str = "Liner Model Conditional Deseasonalizer Detrender"
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        deseason_type = (
            ["additive", "multiplicative"] if self.strictly_positive else ["additive"]
        )
        if self.has_exogeneous_data:
            param_grid = {
                "scaler_x__passthrough": [True, False],
                "forecaster__deseasonalizer__model": deseason_type,
                # "forecaster__deseasonalizer__sp": [self.sp, 2 * self.sp],
                "forecaster__detrender__forecaster__degree": randint(1, 10),
                "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
                "forecaster__reducer__estimator__fit_intercept": [True, False],
            }
        else:
            param_grid = {
                "deseasonalizer__model": deseason_type,
                # "deseasonalizer__sp": [self.sp, 2 * self.sp],
                "detrender__forecaster__degree": randint(1, 10),
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__fit_intercept": [True, False],
            }
        return param_grid
