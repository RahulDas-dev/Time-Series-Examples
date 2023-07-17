from scipy.stats.distributions import randint

from automl.ml_model import MLModel
from automl.basemodel import BaseModel, ModelID, ModelType
from automl.stat.statistics import SeriesStat


class LinearModel(MLModel, BaseModel):
    _identifier: str = ModelID.LinearModelCCD
    _description: str = "Liner Model Conditional Deseasonalizer Detrender"
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    def get_regressors(self):
        try:
            from sklearnex.linear_model import LinearRegression
        except ImportError:
            from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor_args = self.find_regressor_config(regressor)
        return LinearRegression(**regressor_args)

    @property
    def hyper_parameters(self):
        deseason_type = (
            ["additive", "multiplicative"] if self.strictly_positive else ["additive"]
        )
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__deseasonalizer__model": deseason_type,
            # "forecaster__deseasonalizer__sp": [self.sp, 2 * self.sp],
            "forecaster__detrender__forecaster__degree": randint(1, 10),
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            # "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid
