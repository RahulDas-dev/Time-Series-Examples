# decision_tree_regressor.py

from scipy.stats.distributions import loguniform, randint

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_model import MLPipelineCCD, MLPipelineSimple
from automl.stat.statistics import SeriesStat


class DecisionTreeModel(MLPipelineSimple, BaseModel):
    _identifier: str = ModelID.DecisionTree
    _description: str = "Decision Tree Regressor"
    _mtype: ModelType = ModelType.TREE_BASED_MODEL

    def __init__(self, stat: SeriesStat):
        super().__init__(stat)

    @property
    def hyper_parameters(self):
        param_grid = {
            "scaler_x__passthrough": [True, False],
            "forecaster__reducer__window_length": randint(self.sp, 2 * self.sp),
            "forecaster__reducer__estimator__criterion": [
                "mse",
                "friedman_mse",
                "mae",
            ],
            "forecaster__reducer__estimator__min_impurity_decrease": loguniform(
                lower=0.000000001, upper=0.5
            ),
            "forecaster__reducer__estimator__max_depth": randint(1, 10),
            "forecaster__reducer__estimator__min_samples_split": randint(2, 10),
            "forecaster__reducer__estimator__min_samples_leaf": randint(2, 6),
        }
        return param_grid

    def get_regressors(self):
        from sklearn.tree import DecisionTreeRegressor

        regressor_args = self.find_regressor_config(DecisionTreeRegressor())
        return DecisionTreeRegressor(**regressor_args)


class DecisionTreeCCD(MLPipelineCCD, BaseModel):
    _identifier: str = ModelID.DecisionTreeCCD
    _description: str = "Decision Tree Regressor Conditional Deseasonalizer Detrender"
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
            "forecaster__reducer__estimator__criterion": [
                "mse",
                "friedman_mse",
                "mae",
            ],
            "forecaster__reducer__estimator__min_impurity_decrease": loguniform(
                lower=0.000000001, upper=0.5
            ),
            "forecaster__reducer__estimator__max_depth": randint(1, 10),
            "forecaster__reducer__estimator__min_samples_split": randint(2, 10),
            "forecaster__reducer__estimator__min_samples_leaf": randint(2, 6),
        }
        return param_grid

    def get_regressors(self):
        from sklearn.tree import DecisionTreeRegressor

        rregressor_args = self.find_regressor_config(DecisionTreeRegressor())
        return DecisionTreeRegressor(**rregressor_args)
