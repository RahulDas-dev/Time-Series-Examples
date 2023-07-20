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
    BayesianRidge = 11
    BayesianRidgeCCD = 12
    HuberRegressor = 13
    HuberRegressorCCD = 14
    KNeighborsRegressor = 15
    KNeighborsRegressorCCD = 16
    DecisionTree = 20
    DecisionTreeCCD = 21
    RandomForest = 22
    RandomForestCCD = 23
    XGBoost = 24
    XGBoostCCD = 25
    LightGBM = 26
    LightGBMCCD = 27


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
