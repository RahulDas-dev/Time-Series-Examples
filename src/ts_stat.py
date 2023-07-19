import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pmdarima.arima.utils import ndiffs, nsdiffs
from sktime.param_est.seasonality import SeasonalityACF
from sktime.transformations.series.difference import Differencer
from sktime.utils.seasonality import \
    autocorrelation_seasonality_test as acf_sp_test
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesStat:
    frequency: str
    is_strickly_positive: bool
    is_white_noise: bool
    is_seasonal: bool
    seasonality_type: Union[str, None]
    primary_seasonality: int
    candidate_sps: List[int]
    significant_sps: List[int]
    all_sps_to_use: List[int]
    lower_d: int
    uppercase_d: int

    def __repr__(self):
        fields = "\n\t".join(
            f"{fld} = {getattr(self, fld)!r}" for fld in self.__annotations__
        )
        return f"{self.__class__.__name__}[\n\t{fields} \n]"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dataframe(self):
        return pd.DataFrame.from_dict(asdict(self), orient="index", columns=["Values"])


class ExtractStats:
    MAX_SP: int = 60
    NO_SP_2_USE: int = 1
    SP_DETECTION_ALGO: str = "AUTO"  # 'INDEX'

    def __init__(self, frequency: str):
        self.__frequency = frequency
        self.__is_strickly_positive = False
        self.__is_white_noise = False
        self.__is_seasonal = False
        self.__seasonality_type = None
        self.__primary_sp = 1
        self.__candidate_sps = []
        self.__significant_sps = []
        self.__all_sps_to_use = []
        self.__lowercase_d = 0
        self.__uppercase_d = 0

    def extract_statistics(self, data: Union[np.ndarray, pd.Series]) -> SeriesStat:
        data_ = data.copy(deep=True)
        (
            self.__detect_is_strickly_positive(data_)
            .__detect_seasonality_degree(data_)
            .__detect_seasonality_type(data_)
            .__detect_lowr_d(data_)
            .__detect_upper_d(data_)
        )
        return SeriesStat(
            self.__frequency,
            self.__is_strickly_positive,
            self.__is_white_noise,
            self.__is_seasonal,
            self.__seasonality_type,
            self.__primary_sp,
            self.__candidate_sps,
            self.__significant_sps,
            self.__all_sps_to_use,
            self.__lowercase_d,
            self.__uppercase_d,
        )

    def __detect_is_strickly_positive(self, data: Union[np.ndarray, pd.Series]):
        logger.info("detecting strickly positive data...")
        strictly_positive = np.all(data > 0)
        logger.info(f"detecting strickly positive data {strictly_positive}")
        self.__is_strickly_positive = strictly_positive
        return self

    def __detect_seasonality_periods(self, data: Union[np.ndarray, pd.Series]):
        data_t = data.copy(deep=True)
        logger.info("detecting seasonality periods...")
        for i in np.arange(ndiffs(data_t)):
            logger.info(f"Differencing: {i+1}")
            differencer = Differencer()
            data_t = differencer.fit_transform(data_t)
        nobs = len(data_t)
        lags_to_use = int((nobs - 1) / 2)
        sp_est = SeasonalityACF(nlags=lags_to_use)
        sp_est.fit(data_t)

        primary_sp = sp_est.get_fitted_params().get("sp")
        significant_sps = sp_est.get_fitted_params().get("sp_significant")
        if isinstance(significant_sps, np.ndarray):
            significant_sps = significant_sps.tolist()

        logger.info(f"Lags used for seasonal detection: {lags_to_use}")
        logger.info(
            f"Detected Significant SP: {significant_sps[:3]} ... {significant_sps[-3:]}"
        )
        logger.info(f"Detected Primary SP: {primary_sp}")

        return primary_sp, significant_sps, lags_to_use

    def __detect_seasonality_degree(self, data: Union[np.ndarray, pd.Series]):
        logger.info("detecting seasonality degree ...")

        data_t = data.copy(deep=True)
        candidate_sps = None
        skip_autocorrelation_test = False

        if self.SP_DETECTION_ALGO == "AUTO":
            _, candidate_sps, _ = self.__detect_seasonality_periods(data_t)
            skip_autocorrelation_test = True
        elif self.SP_DETECTION_ALGO == "INDEX":
            # candidate_sps = [data.index.freqstr]
            raise NotImplementedError
        else:
            raise ValueError(f"SP_DETECTION_ALGO is invalid {self.SP_DETECTION_ALGO}")

        candidate_sps = [sp for sp in candidate_sps if sp <= self.MAX_SP]

        if skip_autocorrelation_test:
            sp_test_results = [True for sp in candidate_sps]
        else:
            data_tt = data.copy(deep=True)
            sp_test_results = [acf_sp_test(data_tt, sp) for sp in candidate_sps]

        seasonality_present = any(sp_test_results)

        significant_sps = [
            sp for sp, sp_present in zip(candidate_sps, sp_test_results) if sp_present
        ] or [1]

        all_sps_to_use = significant_sps.copy()
        if self.NO_SP_2_USE > 0:
            if len(all_sps_to_use) > self.NO_SP_2_USE:
                all_sps_to_use = all_sps_to_use[: self.NO_SP_2_USE]

        logger.info(f"is_seasonal       {seasonality_present}")
        logger.info(f"primary_sp_2_use  {all_sps_to_use[0]}")
        logger.info(f"candidate_sps     {candidate_sps}")
        logger.info(f"ignificant_sps    {significant_sps}")
        logger.info(f"all_sps_to_use    {all_sps_to_use}")

        self.__is_seasonal = seasonality_present
        self.__primary_sp = all_sps_to_use[0]
        self.__candidate_sps = candidate_sps
        self.__significant_sps = significant_sps
        self.__all_sps_to_use = all_sps_to_use
        return self

    def __detect_seasonality_type(self, data: Union[np.ndarray, pd.Series]):
        # is_strickly_positive =  cls.detect_is_strickly_positive(data)
        # seasonality_present, primary_sp, _, _  = cls.detect_seasonality(data)
        logger.info("detecting seasonality Type ...")
        seasonality_type = None
        if self.__is_seasonal is False:
            seasonality_type = None
        elif self.__is_seasonal and (not self.__is_strickly_positive):
            seasonality_type = "Additive"
        elif self.__is_seasonal and self.__is_strickly_positive:
            decomp_add = seasonal_decompose(
                data, period=self.__primary_sp, model="additive"
            )
            decomp_mult = seasonal_decompose(
                data, period=self.__primary_sp, model="multiplicative"
            )
            if decomp_add is None or decomp_mult is None:
                seasonality_type = "Multiplicative"
            else:
                var_r_add = (np.std(decomp_add.resid)) ** 2
                var_rs_add = (np.std(decomp_add.resid + decomp_add.seasonal)) ** 2
                var_r_mult = (np.std(decomp_mult.resid)) ** 2
                var_rs_mult = (np.std(decomp_mult.resid * decomp_mult.seasonal)) ** 2

                Fs_add = np.maximum(1 - var_r_add / var_rs_add, 0)
                Fs_mult = np.maximum(1 - var_r_mult / var_rs_mult, 0)

                if Fs_mult > Fs_add:
                    seasonality_type = "Multiplicative"
                else:
                    seasonality_type = "Additive"
        else:
            seasonality_type = None
        self.__seasonality_type = seasonality_type
        logger.info(" seasonality Type {seasonality_type}")
        return self

    def __detect_lowr_d(self, data: Union[np.ndarray, pd.Series]):
        logger.info("detecting lowercase_d ...")
        self.__lowercase_d = ndiffs(data)
        logger.info(f"lowercase_d ...{self.__lowercase_d}")
        return self

    def __detect_upper_d(self, data: Union[np.ndarray, pd.Series]):
        # recommended_uppercase_d = nsdiffs(data, m=self.primary_sp_2_use)
        logger.info("detecting upper_d ...")
        if self.__primary_sp > 1:
            try:
                max_D = 2
                uppercase_d = nsdiffs(x=data, m=self.__primary_sp, max_D=max_D)
            except ValueError:
                logger.info("Test for computing 'D' failed at max_D = 2.")
                try:
                    max_D = 1
                    uppercase_d = nsdiffs(x=data, m=self.__primary_sp, max_D=max_D)
                except ValueError:
                    logger.info("Test for computing 'D' failed at max_D = 1.")
                    uppercase_d = 0
        else:
            uppercase_d = 0
        self.__uppercase_d = uppercase_d
        logger.info(f"uppercase_d ...{self.__uppercase_d}")
        return self

    def __remove_harmonics(
        self, significant_sps: list, harmonic_order_method: str = "raw_strength"
    ) -> list:
        """Remove harmonics from the list provided. Similar to Kats - Ref:
        https://github.com/facebookresearch/Kats/blob/v0.2.0/kats/detectors/seasonality.py#L311-L321

        Parameters
        ----------
        significant_sps : list
            The list of significant seasonal periods (ordered by significance)
        harmonic_order_method: str, default = "harmonic_strength"
            This determines how the harmonics are replaced.
            Allowed values are "harmonic_strength", "harmonic_max" or "raw_strength.
            - If set to  "harmonic_strength", then lower seasonal period is replaced by its
            highest strength harmonic seasonal period in same position as the lower seasonal period.
            - If set to  "harmonic_max", then lower seasonal period is replaced by its
            highest harmonic seasonal period in same position as the lower seasonal period.
            - If set to  "raw_strength", then lower seasonal periods is removed and the
            higher harmonic seasonal periods is retained in its original position
            based on its seasonal strength.

            e.g. Assuming detected seasonal periods in strength order are [2, 3, 4, 50]
            and remove_harmonics = True, then:
            - If harmonic_order_method = "harmonic_strength", result = [4, 3, 50]
            - If harmonic_order_method = "harmonic_max", result = [50, 3, 4]
            - If harmonic_order_method = "raw_strength", result = [3, 4, 50]

        Returns
        -------
        list
            The list of significant seasonal periods with harmonics removed
        """
        # Convert period to frequency for harmonic removal
        significant_freqs = [1 / sp for sp in significant_sps]

        if len(significant_freqs) > 1:
            # Sort from lowest freq to highest
            significant_freqs = sorted(significant_freqs)
            # Start from highest freq and remove it if it is a multiple of a lower freq
            # i.e if it is a harmonic of a lower frequency
            for i in range(len(significant_freqs) - 1, 0, -1):
                for j in range(i - 1, -1, -1):
                    fraction = (significant_freqs[i] / significant_freqs[j]) % 1
                    if fraction < 0.001 or fraction > 0.999:
                        significant_freqs.pop(i)
                        break

        # Convert frequency back to period
        # Rounding, else there is precision issues
        filtered_sps = [round(1 / freq, 4) for freq in significant_freqs]

        if harmonic_order_method == "raw_strength":
            # Keep order of significance
            final_filtered_sps = [sp for sp in significant_sps if sp in filtered_sps]
        else:
            # Replace higher strength sp with lower strength harmonic sp
            retained = [True if sp in filtered_sps else False for sp in significant_sps]
            final_filtered_sps = []
            for i, sp_iter in enumerate(significant_sps):
                if retained[i] is False:
                    div = [sp / sp_iter for sp in significant_sps]
                    div_int = [round(elem) for elem in div]
                    equal = [True if a == b else False for a, b in zip(div, div_int)]
                    replacement_candidates = [
                        sp for sp, eq in zip(significant_sps, equal) if eq
                    ]
                    if harmonic_order_method == "harmonic_max":
                        replacement_sp = max(replacement_candidates)
                    elif harmonic_order_method == "harmonic_strength":
                        replacement_sp = replacement_candidates[
                            [
                                i
                                for i, candidate in enumerate(replacement_candidates)
                                if candidate != sp_iter
                            ][0]
                        ]
                    final_filtered_sps.append(replacement_sp)
                else:
                    final_filtered_sps.append(sp_iter)
            # Replacement for ordered set: https://stackoverflow.com/a/53657523/8925915
            final_filtered_sps = list(dict.fromkeys(final_filtered_sps))

        return final_filtered_sps
