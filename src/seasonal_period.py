from enum import IntEnum
from math import gcd
import re


class SeasonalPeriod(IntEnum):
    """ENUM corresponding to Seasonal Periods

    B        business day frequency
    C        custom business day frequency
    D        calendar day frequency
    W        weekly frequency
    M        month end frequency
    SM       semi-month end frequency (15th and end of month)
    BM       business month end frequency
    CBM      custom business month end frequency
    MS       month start frequency
    SMS      semi-month start frequency (1st and 15th)
    BMS      business month start frequency
    CBMS     custom business month start frequency
    Q        quarter end frequency
    BQ       business quarter end frequency
    QS       quarter start frequency
    BQS      business quarter start frequency
    A, Y     year end frequency
    BA, BY   business year end frequency
    AS, YS   year start frequency
    BAS, BYS business year start frequency
    BH       business hour frequency
    H        hourly frequency
    T, min   minutely frequency
    S        secondly frequency
    L, ms    milliseconds
    U, us    microseconds
    N        nanoseconds
    """

    B = 5
    C = 5
    D = 7
    W = 52
    M = 12
    SM = 24
    BM = 12
    CBM = 12
    MS = 12
    SMS = 24
    BMS = 12
    CBMS = 12
    Q = 4
    BQ = 4
    QS = 4
    BQS = 4
    A = 1
    Y = 1
    BA = 1
    BY = 1
    AS = 1
    YS = 1
    BAS = 1
    BYS = 1
    # BH = ??
    H = 24
    T = 60
    min = 60
    S = 60

    @classmethod
    def get_frequency(cls, str_freq: str) -> int:
        """Takes the seasonal period as string detects if it is alphanumeric and returns its integer equivalent.
        For example -
        input - '30W'
        output - 26
        explanation - we take the lcm of 30 and 52 ( as W = 52) which in this case is 780.
        And the output is ( lcm / prefix). Here, 780 / 30 = 26.

        Parameters
        ----------
        str_freq : str
            frequency of the dataset passed as a string

        Returns
        -------
        int
            integer equivalent of the string frequency

        Raises
        ------
        ValueError
            If the frequency suffix does not correspond to any of the values in the
            class SeasonalPeriod then the error is thrown.
        """
        str_freq = str_freq.split("-")[0] or str_freq
        # Checking whether the index_freq contains both digit and alphabet
        if bool(re.search(r"\d", str_freq)):
            temp = re.compile("([0-9]+)([a-zA-Z]+)")
            res = temp.match(str_freq).groups()
            # separating the digits and alphabets
            if res[1] in SeasonalPeriod.__members__:
                prefix = int(res[0])
                value = SeasonalPeriod[res[1]].value
                lcm = abs(value * prefix) // gcd(value, prefix)
                seasonal_period = int(lcm / prefix)
                return seasonal_period
            else:
                raise ValueError(
                    f"Unsupported Period frequency: {str_freq}, valid Period frequency "
                    f"suffixes are: {', '.join(SeasonalPeriod.__members__.keys())}"
                )
        else:
            if str_freq in SeasonalPeriod.__members__:
                seasonal_period = SeasonalPeriod[str_freq].value
                return seasonal_period
            else:
                raise ValueError(
                    f"Unsupported Period frequency: {str_freq}, valid Period frequency "
                    f"suffixes are: {', '.join(SeasonalPeriod.__members__.keys())}"
                )
