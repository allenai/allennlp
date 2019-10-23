from typing import Tuple
import logging

from allennlp.data.iterators.data_iterator import DataIterator

logger = logging.getLogger(__name__)


@DataIterator.register("basic")
class BasicIterator(DataIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """

    def __new__(
        cls,
        batch_size: int = 32,
        instances_per_epoch: int = None,
        max_instances_in_memory: int = None,
        cache_instances: bool = False,
        track_epoch: bool = False,
        maximum_samples_per_batch: Tuple[str, int] = None,
    ) -> None:

        from allennlp.data.iterators.bucket_iterator import BucketIterator

        return BucketIterator(
            batch_size=batch_size,
            instances_per_epoch=instances_per_epoch,
            max_instances_in_memory=max_instances_in_memory,
            cache_instances=cache_instances,
            track_epoch=track_epoch,
            maximum_samples_per_batch=maximum_samples_per_batch,
        )

    def __init__(
        self,
        batch_size: int = 32,
        instances_per_epoch: int = None,
        max_instances_in_memory: int = None,
        cache_instances: bool = False,
        track_epoch: bool = False,
        maximum_samples_per_batch: Tuple[str, int] = None,
    ) -> None:

        pass
