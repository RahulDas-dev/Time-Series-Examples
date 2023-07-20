from scipy.stats.distributions import randint

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class KNeighborsModel(MLPipelineSimple, BaseModel):
    _identifier: str = ModelID.KNeighborsRegressor
    _description: str = "K-Nearest Neighbors Regressor"
    _mtype: ModelType = ModelType.ML_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        if self.has_exogeneous_data:
            param_grid = {
                "scaler_x__passthrough": [True, False],
                "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
                "forecaster__reducer__estimator__n_neighbors": randint(1, 10),
                "forecaster__reducer__estimator__weights": ["uniform", "distance"],
                "forecaster__reducer__estimator__algorithm": [
                    "auto",
                    "ball_tree",
                    "kd_tree",
                    "brute",
                ],
            }
        else:
            param_grid = {
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__n_neighbors": randint(1, 10),
                "reducer__estimator__weights": ["uniform", "distance"],
                "reducer__estimator__algorithm": [
                    "auto",
                    "ball_tree",
                    "kd_tree",
                    "brute",
                ],
            }
        return param_grid

    def get_regressors(self):
        from sklearn.neighbors import KNeighborsRegressor

        regressor_args = self.find_regressor_config(KNeighborsRegressor())
        return KNeighborsRegressor(**regressor_args)


class KNeighborsCCD(MLPipelineCCD, BaseModel):
    _identifier: str = ModelID.KNeighborsRegressorCCD
    _description: str = (
        "K-Nearest Neighbors Regressor Conditional Deseasonalizer Detrender"
    )
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
                "forecaster__reducer__estimator__n_neighbors": randint(1, 10),
                "forecaster__reducer__estimator__weights": ["uniform", "distance"],
                "forecaster__reducer__estimator__algorithm": [
                    "auto",
                    "ball_tree",
                    "kd_tree",
                    "brute",
                ],
            }
        else:
            param_grid = {
                "deseasonalizer__model": deseasonal_type,
                "detrender__forecaster__degree": randint(1, 10),
                "reducer__window_length": randint(self.sp, 2 * self.sp),
                "reducer__estimator__n_neighbors": randint(1, 10),
                "reducer__estimator__weights": ["uniform", "distance"],
                "reducer__estimator__algorithm": [
                    "auto",
                    "ball_tree",
                    "kd_tree",
                    "brute",
                ],
            }
        return param_grid

    def get_regressors(self):
        from sklearn.neighbors import KNeighborsRegressor

        regressor_args = self.find_regressor_config(KNeighborsRegressor())
        return KNeighborsRegressor(**regressor_args)
