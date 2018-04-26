import logging
import random
from typing import Iterable, Dict, List
from overrides import overrides
from allennlp.common.util import is_lazy
from allennlp.data.instance import Instance
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("cached")
class CachedIterator(BucketIterator):
    """
    An extension of the `BucketIterator` that caches its batches between epochs. Batches are built
    only once during the first epoch, then the same batches are used again in consecutive epochs.
    Using this iterator saves run time but costs more memory to store the cached batches. It should
    be used only if the dataset can fit in memory. This iterator shouldn't be used with the lazy
    mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._max_instances_in_memory:
            raise ConfigurationError('max_instances_in_memory should not be set for CachedIterator. '
                                     'Memory needs to be big enough to cache the whole dataset.')
        self.cached_batches: Dict[int, List] = {}

    @overrides
    def _yield_one_epoch(self, instances: Iterable[Instance], shuffle: bool, cuda_device: int, for_training: bool):
        if is_lazy(instances):
            raise ConfigurationError('CachedIterator does not work with lazy dataset readers.')

        instances_id = id(instances)
        if instances_id in self.cached_batches:
            logger.info('returning cached batches of instances id: %d', instances_id)
            batches = self.cached_batches[instances_id]
            if shuffle:
                random.shuffle(batches)  # shuffle the list of batches but not the rows of each batch
            for batch in batches:
                yield batch
        else:
            self.cached_batches[instances_id] = []
            logger.info('caching batches of instances id: %d', instances_id)
            for batch in super()._yield_one_epoch(instances, shuffle, cuda_device, for_training):
                self.cached_batches[instances_id].append(batch)
                yield batch
