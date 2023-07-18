from scipy.stats.distributions import randint, uniform

from automl.models.ml_model import MLModel
from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.stat.statistics import SeriesStat


class ElasticNetModel(MLModel, BaseModel):
    _identifier: str = ModelID.ElasticnetCCD
    _description: str = "ElasticNet Conditional Deseasonalizer Detrender"
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
            "forecaster__reducer__estimator__alpha": uniform(0, 1),
            "forecaster__reducer__estimator__l1_ratio": uniform(0.01, 0.9999999999),
            "forecaster__reducer__estimator__max_iter": [1000],
            # "forecaster__reducer__estimator__fit_intercept": [True, False],
        }
        return param_grid
