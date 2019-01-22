from typing import Iterable, Iterator
import logging

from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("language_modeling")
class LanguageModelingIterator(DataIterator):
    """
    An iterator used for language modeling of contiguous text.
    This is essentially the same as the BasicIterator, but shuffling
    is turned off, the batch size is set to 1, and maximum_samples_per_batch
    is not set.

    Parameters
    ----------
    instances_per_epoch : ``int``, optional, (default = None)
        If specified, each epoch will consist of precisely this many instances.
        If not specified, each epoch will consist of a single pass through the dataset.
    max_instances_in_memory : ``int``, optional, (default = None)
        If specified, the iterator will load this many instances at a time into an
        in-memory list and then produce batches from one such list at a time. This
        could be useful if your instances are read lazily from disk.
    cache_instances : ``bool``, optional, (default = False)
        If true, the iterator will cache the tensorized instances in memory.
        If false, it will do the tensorization anew each iteration.
    track_epoch : ``bool``, optional, (default = False)
        If true, each instance will get a ``MetadataField`` containing the epoch number.
    """

    def __init__(self,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False) -> None:
        super().__init__(batch_size=1,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         maximum_samples_per_batch=None)

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:
        # Set shuffle to False it is True
        if shuffle:
            logger.info("LanguageModelingIterator does not shuffle instances.")
            shuffle = False
        for tensor_dict in super().__call__(instances=instances,
                                            num_epochs=num_epochs,
                                            shuffle=shuffle):
            # Remove singleton dimensions from tensor dict produced
            # by instances generated by LanguageModelingReader
            tensor_dict["inputs"] = tensor_dict["inputs"].squeeze(0)
            tensor_dict["targets"] = tensor_dict["targets"].squeeze(0)
            yield tensor_dict


    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # Set shuffle to False it is True
        if shuffle:
            logger.info("LanguageModelingIterator does not shuffle instances.")
            shuffle = False
        yield from super()._create_batches(instances=instances,
                                           shuffle=shuffle)
