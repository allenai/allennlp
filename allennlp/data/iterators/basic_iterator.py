from typing import List
import random

from overrides import overrides

from allennlp.common.util import group_by_count
from allennlp.data import Dataset, Instance
from allennlp.data.iterators import DataIterator


class BasicIterator(DataIterator):
    """
    A very basic iterator, which takes a dataset, pads all of it's instances to the maximum lengths
    of the relevant fields across the whole dataset, and yields fixed size batches.

    Parameters
    ----------
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    """
    def __init__(self, batch_size: int = 32):
        self._batch_size = batch_size

    @overrides
    def _create_batches(self, dataset: Dataset, shuffle: bool) -> List[List[Instance]]:
        instances = dataset.instances
        if shuffle:
            random.shuffle(instances)
        grouped_instances = group_by_count(instances, self._batch_size, None)
        # The last group might have not been full, so we check if any of the instances
        # are None, which is how group_by_count pads non-complete batches.
        grouped_instances[-1] = [instance for instance in grouped_instances[-1] if instance is not None]
        return grouped_instances
