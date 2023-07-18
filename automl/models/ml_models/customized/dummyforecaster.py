from sktime.forecasting.base import BaseForecaster
import pandas as pd


class DummyForecaster(BaseForecaster):
    """Dummy Forecaster for initial  pipeline"""

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator use the exogenous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,
    }

    def _fit(self, y, X=None, fh=None):
        self._fh_len = None
        if fh is not None:
            self._fh_len = len(fh)
        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None):
        self.check_is_fitted()
        if fh is not None:
            preds = pd.Series([-99_999] * len(fh))
        elif self._fh_len is not None:
            # fh seen during fit
            preds = pd.Series([-99_999] * self._fh_len)
        else:
            raise ValueError(
                f"{type(self).__name__}: `fh` is unknown. Unable to make predictions."
            )
        return preds
