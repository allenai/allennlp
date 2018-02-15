from typing import Iterable, Dict, Iterator, Optional, List
from collections import deque
import itertools
import logging
import math
import random

from overrides import overrides

from allennlp.common import Params
from allennlp.common.util import ensure_list, is_lazy, lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("simple-partition")
class SimplePartitionIterator(BasicIterator):
    """
    A very basic iterator, which takes a dataset, pads all of its instances to the maximum lengths
    of the relevant fields across the whole dataset, and yields fixed size batches.

    Parameters
    ----------
    partition_key : str
        We will use this key to look at each instance's metadata,
        and (consecutive) instances that have the same value are
        required to be in the same batch.
    instances_per_epoch : int, optional, (default = None)
        If specified, each epoch will consist of precisely this many instances.
        If not specified, each epoch will consist of a single pass through the dataset.
    max_instances_in_memory : int, optional, (default = None)
        If specified, the iterator will load this many instances at a time into an
        in-memory list and then produce batches from one such list at a time. This
        could be useful if your instances are read lazily from disk.
    """
    def __init__(self,
                 partition_key: str,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        # Set batch_size to 1 as a placeholder, but we'll update it soon.
        super().__init__(1, instances_per_epoch, max_instances_in_memory)
        self.partitioner = lambda instance: instance.fields['metadata'].metadata[partition_key]

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        leftovers: List[Instance] = []
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            # Start with any leftover instances from the last memory-sized list.
            instances_to_partition = itertools.chain(leftovers, instance_list)
            partitioned = [list(group)
                           for _, group in itertools.groupby(instances_to_partition, self.partitioner)]

            # Save the last batch, it might end mid-partition-group,
            # so we don't want to yield it up yet.
            leftovers = partitioned.pop()

            for batch_instances in partitioned:
                # Update batch_size
                self._batch_size = len(batch_instances)
                yield Batch(batch_instances)

        # Deal with any remaining instances in leftovers.
        if leftovers:
            yield Batch(leftovers)

    @classmethod
    def from_params(cls, params: Params) -> 'SimplePartitionIterator':
        partition_key = params.pop('partition_key')
        instances_per_epoch = params.pop_int('instances_per_epoch', None)
        max_instances_in_memory = params.pop_int('max_instances_in_memory', None)
        params.assert_empty(cls.__name__)
        return cls(partition_key=partition_key,
                   instances_per_epoch=instances_per_epoch,
                   max_instances_in_memory=max_instances_in_memory)
