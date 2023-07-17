from enum import IntEnum, unique


@unique
class ModelType(IntEnum):
    NV_MODEL = 1
    EC_MODEL = 2
    ML_MODEL = 3
    DL_MODEL = 4


@unique
class ModelID(IntEnum):
    LinearModelCCD = 1
    LassoCCD = 2
    ElasticnetCCD = 3


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
