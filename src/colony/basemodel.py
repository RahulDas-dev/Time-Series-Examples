from abc import ABC, abstractproperty
from enum import IntEnum, UNIQUE, verify
from typing import Dict, Any

from src.ts_stat import SeriesStat
from sktime.forecasting.compose import ForecastingPipeline


@verify(UNIQUE)
class ModelType(IntEnum):
    NV_MODEL = 1
    EC_MODEL = 2
    ML_MODEL = 3
    DL_MODEL = 4


class BaseModel(ABC):
    def __init__(self, stat: SeriesStat) -> None:
        super().__init__()
        self.sp = stat.primary_seasonality
        self.strictly_positive = stat.is_strickly_positive
        self.__description = "Base Model"
        self.__rank = 1
        self.__identifier = "Base Model"
        self.__mtype = ModelType.NV_MODEL

    @property
    def description(self):
        return self.__description

    @property
    def rank(self):
        return self.__rank

    @property
    def identifier(self):
        return self.__identifier

    @property
    def mtype(self):
        return self.__mtype

    @abstractproperty
    def model(self) -> ForecastingPipeline:
        pass

    @abstractproperty
    def hyper_parameters(self) -> Dict[str, Any]:
        pass
