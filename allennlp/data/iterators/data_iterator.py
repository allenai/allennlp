from typing import Dict, List, Generator, Union

import numpy

from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance
from allennlp.common import Params
from allennlp.common.registrable import Registrable


class DataIterator(Registrable):
    """
    An abstract ``DataIterator`` class. ``DataIterators`` must implement __call__, which yields
    batched examples.
    """
    default_implementation = 'bucket'

    def __call__(self,
                 dataset: Dataset,
                 num_epochs: int = None,
                 shuffle: bool = True) -> Generator[Dict[str, Union[numpy.ndarray,
                                                                    Dict[str, numpy.ndarray]]], None, None]:
        """
        Returns a generator that yields batches over the given dataset, forever.

        Parameters
        ----------
        dataset : ``Dataset``
        num_epochs : ``int``, optional (default=``None``)
            How times should we iterate over this dataset?  If ``None``, we will iterate over it
            forever.
        shuffle : ``bool``, optional (default=``True``)
            If ``True``, we will shuffle the instances in ``dataset`` before constructing batches
            and iterating over the data.
        """
        if num_epochs is None:
            while True:
                yield from self._yield_one_epoch(dataset, shuffle)
        else:
            for _ in range(num_epochs):
                yield from self._yield_one_epoch(dataset, shuffle)

    def _yield_one_epoch(self, dataset: Dataset, shuffle: bool):
        grouped_instances = self._create_batches(dataset, shuffle)
        for group in grouped_instances:
            batch = Dataset(group)
            padding_lengths = batch.get_padding_lengths()
            yield batch.as_array_dict(padding_lengths, verbose=False)

    def _create_batches(self, dataset: Dataset, shuffle: bool) -> List[List[Instance]]:
        """
        Actually does the work of batching instances in the ``Dataset`` together.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'DataIterator':
        # TODO(Mark): The adaptive iterator will need a bit of work here,
        # to retrieve the scaling function etc.

        iterator_type = params.pop_choice("type", cls.list_available())
        return cls.by_name(iterator_type).from_params(params)
