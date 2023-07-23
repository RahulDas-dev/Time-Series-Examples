from typing import Dict, List, Optional, Union

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_models.ada_boost import AdaBoostCCD, AdaBoostModel
from automl.models.ml_models.bayesian_ridge import (BayesianRidgeCCD,
                                                    BayesianRidgeModel)
from automl.models.ml_models.catboost_model import CatBoostCCD, CatBoostModel
from automl.models.ml_models.decision_tree import (DecisionTreeCCD,
                                                   DecisionTreeModel)
from automl.models.ml_models.elasticnet import ElasticNetCCD, ElasticNetModel
from automl.models.ml_models.extratree_model import (ExtraTreesCCD,
                                                     ExtraTreesModel)
from automl.models.ml_models.grediant_boosting import (GradientBoostCCD,
                                                       GradientBoostModel)
from automl.models.ml_models.huberregressor import (HuberRegressorCCD,
                                                    HuberRegressorModel)
from automl.models.ml_models.knn_regressors import (KNeighborsCCD,
                                                    KNeighborsModel)
from automl.models.ml_models.lasso_model import LassoCCD, LassoModel
from automl.models.ml_models.lassolars_model import (LassoLarsCCD,
                                                     LassoLarsModel)
from automl.models.ml_models.lightgbm_model import LightGBMCCD, LightGBMModel
from automl.models.ml_models.linear_model import LinearModel, LinearModelCCD
from automl.models.ml_models.random_forest import (RandomForestCCD,
                                                   RandomForestModel)
from automl.models.ml_models.ridge_model import RidgeCCD, RidgeModel
from automl.models.ml_models.xgboost_model import XGBoostCCD, XGBoostModel
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
        CatBoostModel,
        CatBoostCCD,
        ExtraTreesModel,
        ExtraTreesCCD,
        GradientBoostModel,
        GradientBoostCCD,
        AdaBoostModel,
        AdaBoostCCD,
    ]

    @classmethod
    def filter_by_model_id(
        cls, stat: SeriesStat, model_id: Optional[Union[str, List[str]]] = None
    ) -> List[BaseModel]:
        if model_id is None:
            return [model(stat) for model in cls._model_list]
        elif isinstance(model_id, str):
            return [
                model(stat)
                for model in cls._model_list
                if model.identifier.name == model_id
            ]
        elif isinstance(model_id, list):
            return [
                model(stat)
                for model in cls._model_list
                if model.identifier.name in model_id
            ]
        else:
            raise ValueError("Model Id not Valid")

    @classmethod
    def filter_by_model_type(
        cls,
        stat: SeriesStat,
        model_type: Optional[Union[ModelID, List[ModelType]]] = None,
    ) -> List[BaseModel]:
        if model_type is None:
            return [model(stat) for model in cls._model_list]
        elif isinstance(model_type, str):
            return [
                model(stat) for model in cls._model_list if model.mtype == model_type
            ]
        elif isinstance(model_type, list):
            return [model for model in cls._model_list if model.mtype in model_type]
        else:
            raise ValueError("Model Type is not Valid")

    @classmethod
    def get_model_object_by_ID(
        cls, stat: SeriesStat, model_id: ModelID
    ) -> Optional[BaseModel]:
        for model in cls._model_list:
            if model.identifier.name == model_id:
                return model(stat)
        return None

    @classmethod
    def select_model_object(cls, stat: SeriesStat, filter: Dict) -> List[BaseModel]:
        if "ModelType" in filter.keys():
            model_types = (
                filter["ModelType"]
                if isinstance(filter["ModelType"], list)
                else [filter["ModelType"]]
            )
            return cls.filter_by_model_type(stat, model_types)
        elif "ModelId" in filter.keys():
            model_ids = (
                filter["ModelId"]
                if isinstance(filter["ModelId"], list)
                else [filter["ModelId"]]
            )
            return cls.filter_by_model_id(stat, model_ids)
        else:
            return cls.filter_by_model_id(stat)
