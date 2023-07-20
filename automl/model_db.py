from typing import List, Optional, Type, Union

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_models.bayesian_ridge import (BayesianRidgeCCD,
                                                    BayesianRidgeModel)
from automl.models.ml_models.decision_tree import (DecisionTreeCCD,
                                                   DecisionTreeModel)
from automl.models.ml_models.elasticnet import ElasticNetCCD, ElasticNetModel
from automl.models.ml_models.huberregressor import (HuberRegressorCCD,
                                                    HuberRegressorModel)
from automl.models.ml_models.knn_regressors import (KNeighborsCCD,
                                                    KNeighborsModel)
from automl.models.ml_models.lasso import LassoCCD, LassoModel
from automl.models.ml_models.lassolars import LassoLarsCCD, LassoLarsModel
from automl.models.ml_models.lightgbm import LightGBMCCD, LightGBMModel
from automl.models.ml_models.linearmodel import LinearModel, LinearModelCCD
from automl.models.ml_models.random_forest import (RandomForestCCD,
                                                   RandomForestModel)
from automl.models.ml_models.ridge import RidgeCCD, RidgeModel
from automl.models.ml_models.xgboost import XGBoostCCD, XGBoostModel
from automl.stat.statistics import SeriesStat


class ModelQuery:
    _model_list = [
        LinearModel,
        LinearModelCCD,
        LassoModel,
        LassoCCD,
        LassoLarsModel,
        LassoLarsCCD,
        ElasticNetModel,
        ElasticNetCCD,
        RidgeModel,
        RidgeCCD,
        BayesianRidgeModel,
        BayesianRidgeCCD,
        HuberRegressorModel,
        HuberRegressorCCD,
        KNeighborsModel,
        KNeighborsCCD,
        DecisionTreeModel,
        DecisionTreeCCD,
        RandomForestModel,
        RandomForestCCD,
        XGBoostModel,
        XGBoostCCD,
        LightGBMModel,
        LightGBMCCD,
    ]

    @classmethod
    def find_model_class_by_id(
        cls, model_id: Optional[Union[str, List[str]]] = None
    ) -> List[Type[BaseModel]]:
        if model_id is None:
            return cls._model_list
        elif isinstance(model_id, str):
            model_id = [model_id]
            return [model for model in cls._model_list if model.identifier in model_id]
        elif isinstance(model_id, list):
            return [model for model in cls._model_list if model.identifier in model_id]

    @classmethod
    def find_model_class_by_type(
        cls, model_type: Optional[ModelType] = None
    ) -> List[Type[BaseModel]]:
        return [model for model in cls._model_list if model.type == ModelType]

    @classmethod
    def find_all_model_object(
        cls, stat: SeriesStat, model_id: Optional[Union[ModelID, List[ModelID]]] = None
    ) -> List[BaseModel]:
        model_classes = cls.find_model_class_by_id(model_id)
        return [model(stat) for model in model_classes]
