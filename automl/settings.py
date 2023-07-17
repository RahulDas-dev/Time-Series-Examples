from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_dir: str = os.path.abspath("../results")
    model_select_count: int = 3
    cv_split: int = 5
    metric: str = "mae"
    random_search_iter: int = 15

    def __repr__(self):
        fields = "\n\t".join(
            f"{fld} = {getattr(self, fld)!r}" for fld in self.__annotations__
        )
        return f"{self.__class__.__name__}[\n\t{fields} \n]"
