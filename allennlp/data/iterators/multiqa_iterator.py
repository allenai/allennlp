from collections import deque
from typing import Iterable, Deque
import logging
import random
from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("multiqa")
class BasicIterator(DataIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """

    def __init__(self,
                 shuffle: bool = True,
                 batch_size: int = 32) -> None:
        super().__init__(batch_size=batch_size)
        self._shuffle = shuffle
        self._batch_size = batch_size

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if self._shuffle:
                random.shuffle(instance_list)
            iterator = iter(instance_list)
            excess: Deque[Instance] = deque()
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    possibly_smaller_batches = sorted(possibly_smaller_batches, key=lambda x: x.fields['metadata'].metadata['question_id'])
                    batch = Batch(possibly_smaller_batches)
                    yield batch
            if excess:
                excess = sorted(excess,key=lambda x: x.fields['metadata'].metadata['question_id'])
                yield Batch(excess)
