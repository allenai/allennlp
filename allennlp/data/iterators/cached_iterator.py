import logging
import random
from typing import Iterable, Dict, List
from overrides import overrides
from allennlp.data.instance import Instance
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.data_iterator import DataIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("cached")
class CachedIterator(BucketIterator):
    """
    An extension of the `BucketIterator` that caches its batches between epochs. Batches are built
    only once during the first epoch, then the same batches are used again in consecutive epochs.
    Using this iterator saves run time but costs more memory to store the cached batches. It should
    be used only if the dataset can fit in memory. This iterator shouldn't be used with the lazy
    mode.

    Parameters:
    ----------
        See :class:`BucketIterator`.
    """

    @overrides
    def _yield_one_epoch(self, instances: Iterable[Instance], shuffle: bool, cuda_device: int, for_training: bool):
        if not hasattr(self, 'cached_batches'):
            # instances id -> list[batches]
            self.cached_batches: Dict[int, List] = {}  # pylint: disable=attribute-defined-outside-init

        instances_id = id(instances)
        if instances_id in self.cached_batches:
            logger.info('returning cached batches of instances id: %d', instances_id)
            batches = self.cached_batches[instances_id]
            if shuffle:
                random.shuffle(batches)  # shuffle the list of batches but not each batch
            for batch in batches:
                yield batch
        else:
            self.cached_batches[instances_id] = []
            logger.info('caching batches of instances id: %d', instances_id)
            for batch in super()._yield_one_epoch(instances, shuffle, cuda_device, for_training):
                self.cached_batches[instances_id].append(batch)
                yield batch
