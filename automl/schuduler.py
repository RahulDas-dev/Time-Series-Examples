import pandas as pd

from automl.model_selection import ModelSelector
from automl.ts_stat import ExtractStats
from automl.tune_model import ModelTuner


class Schuduler:
    def __init__(
        self,
        y: pd.Series,
        fh: int,
        x: pd.DataFrame = None,
        cv_split: int = 5,
        score: str = "mae",
        frequency: str = "H",
    ):
        self.y = y
        self.x = x
        self.fh = fh
        self.cv_split = cv_split
        self.score = score
        self.statistics = None
        self.frequency = "H"
        self.model_ids = None

    def set_y(self, y: pd.Series):
        self.y = y
        return self

    def set_x(self, x: pd.DataFrame):
        self.x = x
        return self

    def set_fh(self, fh: int):
        self.fh = fh
        return self

    def set_cv_split(self, cv_split: int):
        self.cv_split = cv_split
        return self

    def set_score(self, score: str):
        self.score = score
        return self

    def set_frequency(self, frequency: str):
        self.frequency = frequency
        return self

    def extract_statistics(self):
        self.statistics = ExtractStats(frequency=self.frequency).extract_statistics(
            self.y
        )
        return self

    def select_model(self):
        self.model_ids = ModelSelector(
            self.y, self.fh, self.x, self.cv_split, self.score
        ).select_models(self.statistics)
        return self

    def tune_models(self):
        self.tuned_model = ModelTuner(
            self.y, self.fh, self.x, self.cv_split, self.score
        ).tune_model(self.statistics, self.model_ids)
        return self
