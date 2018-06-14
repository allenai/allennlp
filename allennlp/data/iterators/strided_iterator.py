from typing import Iterable, Dict, Iterator, Optional, List
import logging
import math
import random

from overrides import overrides

from allennlp.common import Params
from allennlp.common.util import ensure_list, is_lazy, lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch
from allennlp.common.checks import ConfigurationError

@DataIterator.register("strided")
class StridedIterator(DataIterator):
    """
    A very basic iterator for the lanaguage modeling task. Its difference from the basic iterator 
    is that it would not construct the batch with continuous sentences, but with sentences with 
    a fixed interval. 

    Parameters
    ----------
    batch_size : int, optional, (default = 20)
        The size of each batch of instances yielded when calling the iterator.
    """
    def __init__(self, batch_size: int = 20, lazy: bool = False) -> None:
        self._batch_size = batch_size
        if lazy:
            raise ConfigurationError(f"This iterator cannot be used lazy")

    @overrides
    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        return math.floor(len(ensure_list(instances)) / self._batch_size)        

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        
        instances = ensure_list(instances)
        instances_len = len(instances)

        num_batches = math.floor(instances_len / self._batch_size)
        
        # want all batches to be the same size
        stop = instances_len - instances_len % self._batch_size

        for batch_ind in range(num_batches):
            yield Batch(instances[batch_ind: stop: num_batches])

    @classmethod
    def from_params(cls, params: Params) -> 'BasicIterator':
        batch_size = params.pop_int('batch_size', 20)
        lazy = params.pop_int('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(batch_size=batch_size, lazy=lazy)