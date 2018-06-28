from typing import Iterable
import logging
import math
import random

from overrides import overrides

from allennlp.common import Params
from allennlp.common.util import ensure_list, is_lazy, lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("basic")
class BasicIterator(DataIterator):
    """
    A very basic iterator, which takes a dataset, creates fixed sized batches,
    and pads all of the instances in a batch to the maximum lengths of the relevant fields
    within that batch.

    Parameters
    ----------
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        If specified, each epoch will consist of precisely this many instances.
        If not specified, each epoch will consist of a single pass through the dataset.
    max_instances_in_memory : int, optional, (default = None)
        If specified, the iterator will load this many instances at a time into an
        in-memory list and then produce batches from one such list at a time. This
        could be useful if your instances are read lazily from disk.
    """
    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False) -> None:
        super().__init__(cache_instances=cache_instances, track_epoch=track_epoch)
        self._batch_size = batch_size
        self._instances_per_epoch = instances_per_epoch
        self._max_instances_in_memory = max_instances_in_memory

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)
            iterator = iter(instance_list)
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                batch = Batch(batch_instances)
                yield batch

    @classmethod
    def from_params(cls, params: Params) -> 'BasicIterator':
        batch_size = params.pop_int('batch_size', 32)
        instances_per_epoch = params.pop_int('instances_per_epoch', None)
        max_instances_in_memory = params.pop_int('max_instances_in_memory', None)
        params.assert_empty(cls.__name__)
        return cls(batch_size=batch_size,
                   instances_per_epoch=instances_per_epoch,
                   max_instances_in_memory=max_instances_in_memory)
