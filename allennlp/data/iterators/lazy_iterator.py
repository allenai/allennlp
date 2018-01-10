from typing import List, Iterator, Dict
import itertools
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import LazyDataset
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DataIterator.register("lazy")
class LazyIterator(DataIterator[LazyDataset]):
    """
    A very basic lazy iterator, which takes a dataset and yields fixed size batches.
    Each batch is padded to the maximum length within that batch.

    Parameters
    ----------
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch: int, optional, (default = None)
        If specified, each call to the iterator only ranges over part of the dataset,
        storing its position between calls.
    """
    def __init__(self, batch_size: int = 32, instances_per_epoch: int = None) -> None:
        self._batch_size = batch_size
        self._instances_per_epoch = instances_per_epoch

        # As you might use this iterator with multiple datasets,
        # you need to store a cursor for each one.
        self._cursors: Dict[LazyDataset, Iterator[Instance]] = {}

    @overrides
    def get_num_batches(self, _: LazyDataset) -> int:
        # pylint: disable=no-self-use
        # TODO(joelgrus): figure out the right way to handle this
        return 1

    def _one_epoch(self, dataset: LazyDataset) -> Iterator[Instance]:
        """
        Return one epoch worth of instances.
        """
        if self._instances_per_epoch is None:
            # Just iterate over the whole
            yield from dataset
            return

        # Otherwise we only want the next `instances_per_epoch` items.

        # If we don't have a cursor for this dataset, create one.
        iterator = self._cursors.get(dataset, iter(dataset))

        # Check if the iterator is exhausted:
        try:
            instance = next(iterator)
            # Not exhausted, so yield the instance and bump the start index.
            yield instance
            start_idx = 1
        except StopIteration:
            # Exhausted, so reset the iterator and start at 0
            iterator = iter(dataset)
            start_idx = 0

        yield from (next(iterator) for _ in range(start_idx, self._instances_per_epoch))

        # And now store the iterator, which may be new.
        self._cursors[dataset] = iterator

    @overrides
    def _create_batches(self, dataset: LazyDataset, shuffle: bool) -> Iterator[List[Instance]]:
        if shuffle:
            # TODO(joelgrus): figure out how to configure this and then raise ConfigurationError
            logger.warning("cannot shuffle a lazy dataset")

        # Get an iterator for the next epoch and batch it.
        iterator = self._one_epoch(dataset)

        while True:
            instances = list(itertools.islice(iterator, self._batch_size))
            if instances:
                yield instances
            else:
                return

    @classmethod
    def from_params(cls, params: Params) -> 'LazyIterator':
        batch_size = params.pop_int('batch_size', 32)
        instances_per_epoch = params.pop_int('instances_per_epoch', None)
        params.assert_empty(cls.__name__)
        return cls(batch_size=batch_size, instances_per_epoch=instances_per_epoch)
