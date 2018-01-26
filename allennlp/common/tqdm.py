"""
This wraps tqdm so we can add configurable global defaults for certain tqdm parameters.
"""
from tqdm import tqdm as _tqdm

class Tqdm:
    default_mininterval: int = 1

    @staticmethod
    def set_default_mininterval(value: int) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {
            **kwargs,
            'mininterval': Tqdm.default_mininterval
        }

        return _tqdm(*args, **new_kwargs)
