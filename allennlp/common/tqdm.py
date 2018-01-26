"""
This wraps tqdm so we can add configurable global defaults for certain tqdm parameters.
"""
from tqdm import tqdm as _tqdm

class Tqdm:
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {
                'mininterval': Tqdm.default_mininterval,
                **kwargs
        }

        return _tqdm(*args, **new_kwargs)
