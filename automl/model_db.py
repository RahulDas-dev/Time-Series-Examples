from typing import List, Optional, Type, Union

from automl.models.basemodel import BaseModel, ModelID, ModelType
from automl.models.ml_models.elasticnet import ElasticNetCCD, ElasticNetModel
from automl.models.ml_models.lasso import LassoCCD, LassoModel
from automl.models.ml_models.lassolars import LassoLarsModel, LassoLarsCCD
from automl.models.ml_models.linearmodel import (LinearModelCCD,
                                                 LinearModel)
from automl.models.ml_models.ridge import RidgeModel, RidgeCCD
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
