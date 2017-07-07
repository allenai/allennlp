from typing import Dict, List, Generator

import numpy

from allennlp.data import Dataset
from allennlp.common import Params


class DataIterator:
    """
    An abstract ``DataIterator`` class. ``DataIterators`` must implement __call__, which yields
    batched examples.
    """
    def __call__(self, dataset: Dataset) -> Generator[Dict[str, List[numpy.array]], None, None]:
        """
        Returns a generator that yields batches over the given dataset, forever.

        Parameters
        ----------
        dataset : ``Dataset``
        """
        raise NotImplementedError

    def yield_one_pass(self, dataset: Dataset) -> Generator[Dict[str, List[numpy.array]], None, None]:
        """
        Returns a generator that yields batches over the given dataset, stopping after the dataset
        has been passed over exactly once.

        Parameters
        ----------
        dataset : ``Dataset``
        """

    def num_batches_per_epoch(self, dataset: Dataset) -> int:
        """
        Returns the number of batches there are in a dataset for a single epoch.  If you want to
        set learning rates, etc., based on number of passes over the dataset, this is how you get
        the number of batches to do per epoch.  If you just want to be sure that you only go over a
        dataset once, e.g., for validation or test data, you can just use :func:`yield_one_pass`.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        from . import iterators
        # TODO(Mark): The adaptive iterator will need a bit of work here,
        # to retrieve the scaling function etc.
        iterator_type = params.pop_choice("type", iterators.keys())
        return iterators[iterator_type](**params.as_dict())
