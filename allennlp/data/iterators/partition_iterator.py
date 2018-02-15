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


@DataIterator.register("partition")
class PartitionIterator(BasicIterator):
    """
    A very basic iterator, which takes a dataset, pads all of its instances to the maximum lengths
    of the relevant fields across the whole dataset, and yields fixed size batches.

    Parameters
    ----------
    partition_key : str
        We will use this key to look at each instance's metadata,
        and (consecutive) instances that have the same value are
        required to be in the same batch.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
        Because of the `partition_key` criterion, each batch may be slightly bigger.
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
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        super().__init__(batch_size, instances_per_epoch, max_instances_in_memory)
        self.partitioner = lambda instance: instance.fields['metadata'].metadata[partition_key]

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        batch_instances: List[Instance] = []
        leftovers: List[Instance] = []
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            # Start with any leftover instances from the last memory-sized list.
            instances_to_partition = itertools.chain(leftovers, instance_list)
            partitioned = deque(list(group)
                                for _, group in itertools.groupby(instances_to_partition, self.partitioner))

            # Save the last batch, it might end mid-partition-group,
            # so we don't want to yield it up yet.
            leftovers = partitioned.pop()

            # As long as we have non-last-batch groups,
            # add them to batch_instances. Whenever batch_instances
            # exceeds ``self._batch_size``, yield that batch.
            while partitioned:
                batch_instances.extend(partitioned.popleft())

                if len(batch_instances) >= self._batch_size:
                    yield Batch(batch_instances)
                    batch_instances = []

            # Batch instances might not be empty here, but
            # that's OK, we'll continue with it next iteration.

        # Deal with any remaining instances in batch_instances or leftovers.
        batch_instances.extend(leftovers)
        if batch_instances:
            yield Batch(batch_instances)

    @classmethod
    def from_params(cls, params: Params) -> 'PartitionIterator':
        partition_key = params.pop('partition_key')
        batch_size = params.pop_int('batch_size', 32)
        instances_per_epoch = params.pop_int('instances_per_epoch', None)
        max_instances_in_memory = params.pop_int('max_instances_in_memory', None)
        params.assert_empty(cls.__name__)
        return cls(partition_key=partition_key,
                   batch_size=batch_size,
                   instances_per_epoch=instances_per_epoch,
                   max_instances_in_memory=max_instances_in_memory)
