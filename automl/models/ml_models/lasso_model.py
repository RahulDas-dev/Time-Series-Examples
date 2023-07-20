from scipy.stats.distributions import randint, uniform

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class Lasso:
    def get_regressors(self):
        try:
            from sklearnex.linear_model import Lasso
        except ImportError:
            from sklearn.linear_model import Lasso
        regressor = Lasso()
        regressor_args = self.find_regressor_config(regressor)
        return Lasso(**regressor_args)


class LassoModel(MLPipelineSimple, BaseModel, Lasso):
    _identifier: str = ModelID.Lasso
    _description: str = "Lasso Model"
    _mtype: ModelType = ModelType.LINEAR_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__alpha": uniform(0.001, 10),
            "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid


class LassoCCD(MLPipelineCCD, BaseModel, Lasso):
    _identifier: str = ModelID.LassoCCD
    _description: str = "Lasso Model Conditional Deseasonalizer Detrender"
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
            "forecaster__reducer__estimator__alpha": uniform(0.001, 10),
            "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid
