from typing import Dict, List, Generator

import numpy

from ..dataset import Dataset
from ...common.params import Params


class Iterator:
    """
    An abstract Iterator class. Iterators must implement __call__, which yields
    batched examples. Additionally, it must specify a private method expressing
    how a dataset is transformed into batches.
    """
    def __call__(self, dataset: Dataset) -> Generator[Dict[str, List[numpy.array]], None, None]:

        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        from . import iterators
        # TODO(Mark): The adaptive iterator will need a bit of work here,
        # to retrieve the scaling function etc.
        iterator_type = params.pop_choice("type", iterators.keys())
        return iterators[iterator_type](**params.as_dict())
