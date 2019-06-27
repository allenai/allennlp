from typing import Dict, Iterable, Iterator, Union
import logging

import torch

from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _remove_batch_dim(singleton: Union[TensorDict, torch.Tensor]) -> TensorDict:
    """Recursively removes the batch dimension from tensors in a TensorDict."""
    if isinstance(singleton, dict):
        return {key: _remove_batch_dim(value) for key, value in singleton.items()}
    elif isinstance(singleton, torch.Tensor):
        return singleton.squeeze(0)
    # TODO(rloganiv): Not sure if this is appropriate for Fields whose as_tensor and batch_tensor
    # methods do not return DataArrays (e.g. MetadataField and ProductionRuleField).
    else:
        return singleton


@DataIterator.register("pass_through")
class PassThroughIterator(BasicIterator):
    """
    An iterator which performs no batching or shuffling of instances, only tensorization. E.g,
    instances are effectively passed 'straight through' the iterator.

    This is essentially the same as a BasicIterator with shuffling disabled, the batch size set
    to 1, and maximum sampled per batch disabled. The only difference is that this iterator
    removes the batch dimension. This can be useful for rare situations where batching is best
    performed within the dataset reader (e.g. for continguous language modeling, or for other
    problems where state is shared across batches).

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
                 shuffle: bool = False) -> Iterator[TensorDict]:
        if shuffle:
            logger.warning("PassThroughIterator does not shuffle instances. If shuffling is "
                           "required, please implement in your DatasetReader.")
            shuffle = False
        for tensor_dict in super().__call__(instances, num_epochs, shuffle):
            yield _remove_batch_dim(tensor_dict)

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        if shuffle:
            logger.warning("PassThroughIterator does not shuffle instances. If shuffling is "
                           "required, please implement in your DatasetReader.")
            shuffle = False
        yield from super()._create_batches(instances, shuffle)
