# extra_trees_regressor.py

from scipy.stats.distributions import randint, uniform

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class ExtraTreesModel(MLPipelineSimple, BaseModel):
    _identifier: str = ModelID.ExtraTrees
    _description: str = "Extra Trees Regressor"
    _mtype: ModelType = ModelType.TREE_BASED_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__n_estimators": randint(50, 150),
            "forecaster__reducer__estimator__criterion": ["mse", "mae"],
            "forecaster__reducer__estimator__max_depth": randint(1, 10),
            "forecaster__reducer__estimator__min_samples_split": randint(2, 20),
            "forecaster__reducer__estimator__min_samples_leaf": randint(1, 20),
            "forecaster__reducer__estimator__max_features": uniform(0.1, 1.0),
        }
        return param_grid

    def get_regressors(self):
        from sklearn.ensemble import ExtraTreesRegressor

        regressor_args = self.find_regressor_config(ExtraTreesRegressor())
        return ExtraTreesRegressor(**regressor_args)


class ExtraTreesCCD(MLPipelineCCD, BaseModel):
    _identifier: str = ModelID.ExtraTreesCCD
    _description: str = "Extra Trees Regressor Conditional Deseasonalizer Detrender"
    _mtype: ModelType = ModelType.TREE_BASED_MODEL

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
            "forecaster__reducer__estimator__criterion": ["mse", "mae"],
            "forecaster__reducer__estimator__max_depth": randint(1, 10),
            "forecaster__reducer__estimator__min_samples_split": randint(2, 20),
            "forecaster__reducer__estimator__min_samples_leaf": randint(1, 20),
            "forecaster__reducer__estimator__max_features": uniform(0.1, 1.0),
        }
        return param_grid

    def get_regressors(self):
        from sklearn.ensemble import ExtraTreesRegressor

        regressor_args = self.find_regressor_config(ExtraTreesRegressor())
        return ExtraTreesRegressor(**regressor_args)
