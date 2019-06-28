from typing import Iterable, Iterator
import itertools
import logging

from overrides import overrides

from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("pass_through")
class PassThroughIterator(DataIterator):
    """
    An iterator which performs no batching or shuffling of instances, only tensorization. E.g,
    instances are effectively passed 'straight through' the iterator.

    This is essentially the same as a BasicIterator with shuffling disabled, the batch size set
    to 1, and maximum samples per batch disabled. The only difference is that this iterator
    removes the batch dimension. This can be useful for rare situations where batching is best
    performed within the dataset reader (e.g. for contiguous language modeling, or for other
    problems where state is shared across batches).
    """
    def __init__(self):
        super().__init__(batch_size=1)

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        raise RuntimeError("PassThroughIterator doesn't use create_batches")

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = False) -> Iterator[TensorDict]:
        # Warn users that this iterator does not do anything for you.
        if shuffle:
            logger.warning("PassThroughIterator does not shuffle instances. If shuffling is "
                           "required, please implement in your DatasetReader.")

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count()
        else:
            epochs = range(num_epochs)

        for _ in epochs:
            for instance in instances:
                if self.vocab is not None:
                    instance.index_fields(self.vocab)
                yield instance.as_tensor_dict()
