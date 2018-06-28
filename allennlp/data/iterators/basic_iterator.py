from typing import Iterable
import logging
import random

from allennlp.common import Params
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("basic")
class BasicIterator(DataIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """
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
        cache_instances = params.pop_bool('cache_instances', False)
        track_epoch = params.pop_bool('track_epoch', False)

        params.assert_empty(cls.__name__)
        return cls(batch_size=batch_size,
                   instances_per_epoch=instances_per_epoch,
                   max_instances_in_memory=max_instances_in_memory,
                   cache_instances=cache_instances,
                   track_epoch=track_epoch)
