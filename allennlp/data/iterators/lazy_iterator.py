import itertools
import logging
import math
from typing import List

from overrides import overrides

from allennlp.common import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("lazy")
class LazyIterator(DataIterator):
    """
    An iterator for iterating lazily across a dataset, yielding fixed size batches.

    Parameters
    ----------
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    """
    def __init__(self, batch_size: int = 32) -> None:
        self._batch_size = batch_size

    def _yield_one_epoch(self, dataset: Dataset, shuffle: bool, cuda_device: int, for_training: bool):
        # TODO(joelgrus): this should probably throw a ConfigurationError
        # but right now `shuffle` is not even part of the configuration
        if shuffle:
            logger.warning("shuffle is not implemented for LazyIterator")

        indexed_instances = enumerate(instance for instance in dataset)
        for _, group in itertools.groupby(indexed_instances, lambda pair: pair[0] // self._batch_size):
            batch = Dataset([instance for _, instance in group])
            padding_lengths = batch.get_padding_lengths()
            logger.debug("Batch padding lengths: %s", str(padding_lengths))
            logger.debug("Batch size: %d", batch.num_instances)
            yield batch.as_tensor_dict(padding_lengths,
                                       cuda_device=cuda_device,
                                       for_training=for_training)

    def _create_batches(self, dataset: Dataset, shuffle: bool) -> List[List[Instance]]:
        raise RuntimeError("should never be called")

    @overrides
    def get_num_batches(self, dataset: Dataset) -> int:
        return math.ceil(dataset.num_instances / self._batch_size)

    @classmethod
    def from_params(cls, params: Params) -> 'LazyIterator':
        batch_size = params.pop('batch_size', 32)
        params.assert_empty(cls.__name__)
        return cls(batch_size=batch_size)
