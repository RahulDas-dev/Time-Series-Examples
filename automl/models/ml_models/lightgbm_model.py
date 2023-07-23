# lightgbm_regressor.py

from scipy.stats.distributions import loguniform, randint

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
            "forecaster__reducer__window_length": randint(10, self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__num_leaves": randint(2, 256),
            "forecaster__reducer__estimator__n_estimators": randint(10, 300),
            "forecaster__reducer__estimator__learning_rate": loguniform(0.000001, 0.5),
            "forecaster__reducer__estimator__max_depth": randint(1, 10),
            "forecaster__reducer__estimator__min_child_samples": randint(1, 100),
            # "forecaster__reducer__estimator__subsample": uniform(0.5, 1),
            # "forecaster__reducer__estimator__colsample_bytree": uniform(0.5, 1),
            # "forecaster__reducer__bagging_freq": randint(1, 7),
            "forecaster__reducer__estimator__reg_alpha": loguniform(0.0000000001, 10),
            "forecaster__reducer__estimator__reg_lambda": loguniform(0.0000000001, 10),
        }
        return param_grid

    def get_regressors(self):
        from lightgbm import LGBMRegressor

        regressor_args = self.find_regressor_config(LGBMRegressor())
        return LGBMRegressor(**regressor_args, verbose=-100)


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
            "forecaster__deseasonalizer__sp": [self.sp, 2 * self.sp],
            "forecaster__detrender__forecaster__degree": randint(1, 10),
            "forecaster__reducer__window_length": randint(10, self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__num_leaves": randint(2, 256),
            "forecaster__reducer__estimator__n_estimators": randint(10, 300),
            "forecaster__reducer__estimator__learning_rate": loguniform(0.000001, 0.5),
            "forecaster__reducer__estimator__max_depth": randint(1, 10),
            "forecaster__reducer__estimator__min_child_samples": randint(1, 100),
            # "forecaster__reducer__estimator__subsample": uniform(0.5, 1),
            # "forecaster__reducer__estimator__colsample_bytree": uniform(0.5, 1),
            # "forecaster__reducer__bagging_freq": randint(1, 7),
            "forecaster__reducer__estimator__reg_alpha": loguniform(0.0000000001, 10),
            "forecaster__reducer__estimator__reg_lambda": loguniform(0.0000000001, 10),
        }
        return param_grid

    def get_regressors(self):
        from lightgbm import LGBMRegressor

        regressor_args = self.find_regressor_config(LGBMRegressor())
        return LGBMRegressor(**regressor_args, verbose=-100)
