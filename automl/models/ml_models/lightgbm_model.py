# lightgbm_regressor.py

from scipy.stats.distributions import randint, uniform

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class LightGBMModel(MLPipelineSimple, BaseModel):
    _identifier: str = ModelID.LightGBM
    _description: str = "LightGBM Regressor"
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
            "forecaster__reducer__estimator__max_depth": randint(1, 10),
            "forecaster__reducer__estimator__min_child_samples": randint(1, 20),
            "forecaster__reducer__estimator__subsample": uniform(0.5, 1),
            "forecaster__reducer__estimator__colsample_bytree": uniform(0.5, 1),
            "forecaster__reducer__estimator__reg_alpha": uniform(0, 1),
            "forecaster__reducer__estimator__reg_lambda": uniform(0, 1),
        }
        return param_grid

    def get_regressors(self):
        from lightgbm import LGBMRegressor

        regressor_args = self.find_regressor_config(LGBMRegressor())
        return LGBMRegressor(**regressor_args)


class LightGBMCCD(MLPipelineCCD, BaseModel):
    _identifier: str = ModelID.LightGBMCCD
    _description: str = "LightGBM Regressor Conditional Deseasonalizer Detrender"
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
            "forecaster__reducer__estimator__max_depth": randint(1, 10),
            "forecaster__reducer__estimator__min_child_samples": randint(1, 20),
            "forecaster__reducer__estimator__subsample": uniform(0.5, 1),
            "forecaster__reducer__estimator__colsample_bytree": uniform(0.5, 1),
            "forecaster__reducer__estimator__reg_alpha": uniform(0, 1),
            "forecaster__reducer__estimator__reg_lambda": uniform(0, 1),
        }
        return param_grid

    def get_regressors(self):
        from lightgbm import LGBMRegressor

        regressor_args = self.find_regressor_config(LGBMRegressor())
        return LGBMRegressor(**regressor_args)
