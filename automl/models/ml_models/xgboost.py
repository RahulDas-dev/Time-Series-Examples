# xgboost_regressor.py

from scipy.stats.distributions import randint, uniform

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class XGBoostModel(MLPipelineSimple, BaseModel):
    _identifier: str = ModelID.XGBoost
    _description: str = "XGBoost Regressor"
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        if self.has_exogeneous_data:
            param_grid = {
                "scaler_x__passthrough": [True, False],
                "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
                "forecaster__reducer__estimator__n_estimators": randint(50, 150),
                "forecaster__reducer__estimator__learning_rate": uniform(0.01, 0.3),
                "forecaster__reducer__estimator__max_depth": randint(1, 10),
                "forecaster__reducer__estimator__min_child_weight": randint(1, 20),
                "forecaster__reducer__estimator__gamma": uniform(0, 1),
                "forecaster__reducer__estimator__subsample": uniform(0.5, 1),
                "forecaster__reducer__estimator__colsample_bytree": uniform(0.5, 1),
                "forecaster__reducer__estimator__reg_alpha": uniform(0, 1),
                "forecaster__reducer__estimator__reg_lambda": uniform(0, 1),
            }
        else:
            param_grid = {
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__n_estimators": randint(50, 150),
                "reducer__estimator__learning_rate": uniform(0.01, 0.3),
                "reducer__estimator__max_depth": randint(1, 10),
                "reducer__estimator__min_child_weight": randint(1, 20),
                "reducer__estimator__gamma": uniform(0, 1),
                "reducer__estimator__subsample": uniform(0.5, 1),
                "reducer__estimator__colsample_bytree": uniform(0.5, 1),
                "reducer__estimator__reg_alpha": uniform(0, 1),
                "reducer__estimator__reg_lambda": uniform(0, 1),
            }
        return param_grid

    def get_regressors(self):
        from xgboost import XGBRegressor

        regressor_args = self.find_regressor_config(XGBRegressor())
        return XGBRegressor(**regressor_args)


class XGBoostCCD(MLPipelineCCD, BaseModel):
    _identifier: str = ModelID.XGBoostCCD
    _description: str = "XGBoost Regressor Conditional Deseasonalizer Detrender"
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
                "forecaster__reducer__estimator__n_estimators": randint(50, 150),
                "forecaster__reducer__estimator__learning_rate": uniform(0.01, 0.3),
                "forecaster__reducer__estimator__max_depth": randint(1, 10),
                "forecaster__reducer__estimator__min_child_weight": randint(1, 20),
                "forecaster__reducer__estimator__gamma": uniform(0, 1),
                "forecaster__reducer__estimator__subsample": uniform(0.5, 1),
                "forecaster__reducer__estimator__colsample_bytree": uniform(0.5, 1),
                "forecaster__reducer__estimator__reg_alpha": uniform(0, 1),
                "forecaster__reducer__estimator__reg_lambda": uniform(0, 1),
            }
        else:
            param_grid = {
                "deseasonalizer__model": deseasonal_type,
                "detrender__forecaster__degree": randint(1, 10),
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__n_estimators": randint(50, 150),
                "reducer__estimator__learning_rate": uniform(0.01, 0.3),
                "reducer__estimator__max_depth": randint(1, 10),
                "reducer__estimator__min_child_weight": randint(1, 20),
                "reducer__estimator__gamma": uniform(0, 1),
                "reducer__estimator__subsample": uniform(0.5, 1),
                "reducer__estimator__colsample_bytree": uniform(0.5, 1),
                "reducer__estimator__reg_alpha": uniform(0, 1),
                "reducer__estimator__reg_lambda": uniform(0, 1),
            }
        return param_grid

    def get_regressors(self):
        from xgboost import XGBRegressor

        regressor_args = self.find_regressor_config(XGBRegressor())
        return XGBRegressor(**regressor_args)
