from typing import List, Optional, Type, Union

from automl.basemodel import BaseModel, ModelID, ModelType
from automl.models.elasticnet import ElasticNetModel
from automl.models.linearmodel import LinearModel
from automl.ts_stat import SeriesStat

model_list = [
    LinearModel,
    ElasticNetModel,
]


class ModelQuery:
    _model_list = [
        LinearModel,
        ElasticNetModel,
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
