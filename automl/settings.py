import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional


class SchudulerState(IntEnum):
    Init = 0
    SetUP = 1
    Extracting_Stat = 2
    Compare_Model = 3
    HyperParameter_Model = 4
    Saving_Model = 5


@dataclass(frozen=True)
class Settings:
    model_dir: str = os.path.abspath("../results")
    model_select_count: int = 3
    cv_split: int = 5
    metric: str = "mae"
    random_search_iter: int = 15
    filter: Optional[Dict] = None

    def __repr__(self):
        fields = "\n\t".join(
            f"{fld} = {getattr(self, fld)!r}" for fld in self.__annotations__
        )
        return f"{self.__class__.__name__}[\n\t{fields} \n]"
