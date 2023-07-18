import warnings

from sktime.transformations.base import BaseTransformer


class ColumnsGuard(BaseTransformer):
    _tags = {
        "scitype:transform-input": "Series",      # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",     # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",       # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,             # is this an instance-wise transform?
        "capability:inverse_transform": True,     # can the transformer inverse transform?
        "univariate-only": False,                 # can the transformer handle multivariate X?
        "X_inner_mtype": "pd.DataFrame",          # which mtypes do _fit/_predict support for X? # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",                  # which mtypes do _fit/_predict support for y?
        "requires_y": False,                      # does y need to be passed in fit?
        "enforce_index_type": None,               # index type that needs to be enforced in X/y
        "fit_is_empty": False,                    # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": True,         # can estimator handle different X/y index?
        "transform-returns-same-time-index": True,  # does transform return have the same time index as input X
        "skip-inverse-transform": False,           # is inverse-transform skipped when called?
        "capability:unequal_length": True,         # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": True,  # is transform result always guaranteed to be equal length (and series)? #not relevant for transformers that return Primitives in transform-output
        "handles-missing-data": False,              # can estimator handle missing data?
        "capability:missing_values:removes": False,  # is transform result always guaranteed to contain no missing values?
    }

    def __init__(self):
        super(ColumnsGuard, self).__init__()

    def _fit(self, X, y=None):
        self._columns_name = X.columns.tolist()
        return self

    # todo: implement this, mandatory
    def _transform(self, X, y=None):
        if self._columns_name == X.columns.tolist():
            X_transformed = X.copy(deep=True)
        else:
            missing_col = set(self._columns_name) - set(X.columns.tolist())
            extra_col = set(X.columns.tolist()) - set(self._columns_name)
            if missing_col:
                raise ValueError("Missing columns: {missing_col}")
            elif extra_col:
                warnings.warn("Got extra columns: {extra_col}, ignoring")
                X_transformed = X[self.self._columns_name].copy(deep=True)
        return X_transformed

    def _inverse_transform(self, X, y=None):
        X_inv_transformed = X.copy(deep=True)
        return X_inv_transformed

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        params = {}
        return params
