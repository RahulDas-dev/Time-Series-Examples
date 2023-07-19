from enum import IntEnum, unique


@unique
class ModelType(IntEnum):
    NV_MODEL = 1
    EC_MODEL = 2
    ML_MODEL = 3
    DL_MODEL = 4


@unique
class ModelID(IntEnum):
    Linear = 1
    LinearCCD = 2
    Lasso = 3
    LassoCCD = 4
    LassoLars = 5
    LassoLarsCCD = 6
    Ridge = 7
    RidgeCCD = 8
    Elasticnet = 9
    ElasticnetCCD = 10


class BaseModel:
    @classmethod
    @property
    def description(cls) -> str:
        return cls._description

    @classmethod
    @property
    def rank(cls) -> int:
        return cls._identifier.value

    @classmethod
    @property
    def identifier(cls) -> ModelID:
        return cls._identifier

    @classmethod
    @property
    def mtype(cls) -> ModelType:
        return cls._mtype
