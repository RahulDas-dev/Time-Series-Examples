# catboost_regressor.py

from scipy.stats.distributions import randint, uniform

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class CatBoostModel(MLPipelineSimple, BaseModel):
    _identifier: str = ModelID.CatBoost
    _description: str = "CatBoost Regressor"
    _mtype: ModelType = ModelType.BOOSTING_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__n_estimators": randint(50, 150),
            "forecaster__reducer__estimator__learning_rate": uniform(0.01, 0.3),
            "forecaster__reducer__estimator__depth": randint(1, 10),
            "forecaster__reducer__estimator__rsm": uniform(0.5, 1),
            "forecaster__reducer__estimator__l2_leaf_reg": uniform(1, 10),
        }
        return param_grid

    def get_regressors(self):
        from catboost import CatBoostRegressor

        regressor_args = self.find_regressor_config(CatBoostRegressor())
        return CatBoostRegressor(**regressor_args)


class CatBoostCCD(MLPipelineCCD, BaseModel):
    _identifier: str = ModelID.CatBoostCCD
    _description: str = "CatBoost Regressor Conditional Deseasonalizer Detrender"
    _mtype: ModelType = ModelType.BOOSTING_MODEL

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
            "forecaster__reducer__estimator__n_estimators": randint(50, 150),
            "forecaster__reducer__estimator__learning_rate": uniform(0.01, 0.3),
            "forecaster__reducer__estimator__depth": randint(1, 10),
            "forecaster__reducer__estimator__rsm": uniform(0.5, 1),
            "forecaster__reducer__estimator__l2_leaf_reg": uniform(1, 10),
        }
        return param_grid

    def get_regressors(self):
        from catboost import CatBoostRegressor

        regressor_args = self.find_regressor_config(CatBoostRegressor())
        return CatBoostRegressor(**regressor_args)
