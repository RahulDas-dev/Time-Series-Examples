from enum import IntEnum, unique


@unique
class ModelType(IntEnum):
    NAIVE_MODEL = 1
    ECONOMETRIC_MODEL = 2
    LINEAR_MODEL = 3
    TREE_BASED_MODEL = 4
    DISTANCE_BASED_MODEL = 5
    BOOSTING_MODEL = 6
    DL_MODEL = 7


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
    CatBoost = 28
    CatBoostCCD = 29
    ExtraTrees = 30
    ExtraTreesCCD = 31
    GradientBoost = 32
    GradientBoostCCD = 33
    AdaBoost = 33
    AdaBoostCCD = 34


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
