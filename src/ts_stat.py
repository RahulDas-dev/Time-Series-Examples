from dataclasses import dataclass, field

from typing import Union, List

import numpy as np
import pandas as pd
from pmdarima.arima.utils import ndiffs
from sktime.transformations.series.difference import Differencer
from sktime.param_est.seasonality import SeasonalityACF
from sktime.utils.seasonality import autocorrelation_seasonality_test as acf_sp_test
from statsmodels.tsa.seasonal import seasonal_decompose


@dataclass
class SeriesStat:
    frequency: str
    is_strickly_positive: bool
    is_white_noise: bool
    is_stationary: bool
    is_seasonal: bool
    seasonality_type: Union[str, None]
    primary_sp: int
    candidate_sps: List[int] = field(default_factory=list)
    significant_sps: List[int] = field(default_factory=list)
    lower_d: int


class ExtractStats:
    MAX_SP: int = 60
    frequency: str
    is_strickly_positive: bool
    is_white_noise: bool
    is_stationary: bool
    is_seasonal: bool
    seasonality_type: str
    primary_sp: str
    candidate_sps: List[int]
    significant_sps: List[int]

    def __init__(self, frequency: str, size: int, verbose: bool = False):
        self.frequency = frequency
        self.is_strickly_positive = False
        self.is_white_noise = False
        self.is_stationary = False
        self.seasonality_type = None
        self.primary_sp = 1
        self.candidate_sps = []
        self.significant_sps = []
        self.verbose = verbose

    def extract_statistics(self, data: Union[np.ndarray, pd.Series]):
        data_ = data.copy(deep=True)
        (
            self._detect_is_strickly_positive(data_)
            ._detect_seasonality_periods(data_)
            ._detect_seasonality_degree(data_)
            ._detect_seasonality_type(data_)
            ._detect_lowr_d(data_)
        )
        return

    def _detect_is_strickly_positive(self, data: Union[np.ndarray, pd.Series]):
        self.is_strickly_positive = np.all(data > 0)
        return self

    def _detect_seasonality_periods(self, data: Union[np.ndarray, pd.Series]):
        for i in np.arange(ndiffs(data)):
            if self.verbose:
                print(f"Differencing: {i+1}")
            differencer = Differencer()
            data = differencer.fit_transform(data)
        nobs = len(data)
        lags_to_use = int((nobs - 1) / 2)
        sp_est = SeasonalityACF(nlags=lags_to_use)
        sp_est.fit(data)

        primary_sp = sp_est.get_fitted_params().get("sp")
        significant_sps = sp_est.get_fitted_params().get("sp_significant")
        if isinstance(significant_sps, np.ndarray):
            significant_sps = significant_sps.tolist()

        if self.verbose:
            print(f"\tLags used for seasonal detection: {lags_to_use}")
            print(f"\tDetected Significant SP: {significant_sps}")
            print(f"\tDetected Primary SP: {primary_sp}")

        # return primary_sp, significant_sps, lags_to_use
        self.significant_sps = significant_sps
        return self

    def _detect_seasonality_degree(
        self, data: Union[np.ndarray, pd.Series], verbose: bool = False
    ):
        # _, candidate_sps, _ = cls.detect_seasonality_periods(data)

        # return primary_sp, significant_sps, lags_to_use

        candidate_sps = [sp for sp in self.significant_sps if sp <= self.MAX_SP]

        data_ = data.copy(deep=True)
        sp_test_results = [acf_sp_test(data_, sp) for sp in candidate_sps]

        seasonality_present = any(sp_test_results)

        significant_sps = [
            sp for sp, sp_present in zip(candidate_sps, sp_test_results) if sp_present
        ] or [1]

        primary_sp = significant_sps[0]

        self.seasonality_present = seasonality_present
        self.primary_sp = primary_sp
        self.candidate_sps = candidate_sps
        self.significant_sps = significant_sps
        # return seasonality_present, primary_sp, candidate_sps, significant_sps
        return self

    def _detect_seasonality_type(self, data: Union[np.ndarray, pd.Series]):
        # is_strickly_positive =  cls.detect_is_strickly_positive(data)
        # seasonality_present, primary_sp, _, _  = cls.detect_seasonality(data)
        if self.seasonality_present is False:
            self.seasonality_type = None
            return self
        if self.seasonality_present and self.is_strickly_positive is False:
            self.seasonality_type = "Additive"
            return self
        # if seasonality_present and is_strickly_positive :
        decomp_add = seasonal_decompose(data, period=self.primary_sp, model="additive")
        decomp_mult = seasonal_decompose(
            data, period=self.primary_sp, model="multiplicative"
        )
        if decomp_add is None or decomp_mult is None:
            self.seasonality_type = "Multiplicative"
            return self
        var_r_add = (np.std(decomp_add.resid)) ** 2
        var_rs_add = (np.std(decomp_add.resid + decomp_add.seasonal)) ** 2
        var_r_mult = (np.std(decomp_mult.resid)) ** 2
        var_rs_mult = (np.std(decomp_mult.resid * decomp_mult.seasonal)) ** 2

        Fs_add = np.maximum(1 - var_r_add / var_rs_add, 0)
        Fs_mult = np.maximum(1 - var_r_mult / var_rs_mult, 0)

        if Fs_mult > Fs_add:
            self.seasonality_type = "Multiplicative"
        else:
            self.seasonality_type = "Additive"
        return self

    def _detect_lowr_d(self, data: Union[np.ndarray, pd.Series]):
        self.lower_d = ndiffs(data)
        return self
