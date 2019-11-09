from typing import Tuple, Iterable, Deque
import logging
from collections import deque
import random

from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.bucket_iterator import BucketIteratorShim
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)


@DataIterator.register("basic")
class BasicIteratorShim(BucketIteratorShim):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """

    def __init__(
        self,
        batch_size: int = 32,
        instances_per_epoch: int = None,
        max_instances_in_memory: int = None,
        cache_instances: bool = False,
        track_epoch: bool = False,
        maximum_samples_per_batch: Tuple[str, int] = None,
    ):

        from allennlp.data.iterators.bucket_iterator import BucketIteratorShim

        super().__init__(
            batch_size=batch_size,
            instances_per_epoch=instances_per_epoch,
            max_instances_in_memory=max_instances_in_memory,
            cache_instances=cache_instances,
            track_epoch=track_epoch,
            maximum_samples_per_batch=maximum_samples_per_batch,
        )


@DataIterator.register("basic_old")
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
            excess: Deque[Instance] = deque()
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(
                    batch_instances, excess
                ):
                    batch = Batch(possibly_smaller_batches)
                    yield batch
            if excess:
                yield Batch(excess)
