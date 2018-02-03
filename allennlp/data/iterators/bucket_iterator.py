import logging
import random
from typing import List, Tuple, Dict, cast, Iterable

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.util import add_noise_to_dict_values
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.iterators.data_iterator import DataIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("bucket")
class BucketIterator(BasicIterator):
    """
    An iterator which by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).

    Parameters
    ----------
    sorting_keys : List[Tuple[str, str]]
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        For example, ``[("sentence1", "num_tokens"), ("sentence2", "num_tokens"), ("sentence1",
        "num_token_characters")]`` would sort a dataset first by the "num_tokens" of the
        "sentence1" field, then by the "num_tokens" of the "sentence2" field, and finally by the
        "num_token_characters" of the "sentence1" field.  TODO(mattg): we should have some
        documentation somewhere that gives the standard padding keys used by different fields.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.

        Note that if you specify ``max_instances_in_memory``, the first batch will only be the
        biggest from among the first "max instances in memory" instances.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        See :class:`BasicIterator`.
    max_instances_in_memory : int, optional, (default = None)
        See :class:`BasicIterator`.
    """

    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        if not sorting_keys:
            raise ConfigurationError("BucketIterator requires sorting_keys to be specified")

        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        super(BucketIterator, self).__init__(batch_size=batch_size,
                                             instances_per_epoch=instances_per_epoch,
                                             max_instances_in_memory=max_instances_in_memory)

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):
            instance_list = self._sort_by_padding(instance_list,
                                                  self._sorting_keys,
                                                  self._padding_noise)

            grouped_instances = list(super()._create_batches(instance_list, shuffle=False))
            move_to_front = self._biggest_batch_first and len(grouped_instances) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = grouped_instances.pop()
                penultimate_batch = grouped_instances.pop()
            if shuffle:
                random.shuffle(grouped_instances)
            else:
                logger.warning("shuffle parameter is set to False,"
                               " while bucket iterators by definition change the order of your data.")
            if move_to_front:
                grouped_instances.insert(0, penultimate_batch)
                grouped_instances.insert(0, last_batch)

            yield from grouped_instances

    def _sort_by_padding(self,
                         instances: List[Instance],
                         sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                         padding_noise: float = 0.0) -> List[Instance]:
        """
        Sorts the ``Instances`` in this ``Batch`` by their padding lengths, using the keys in
        ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
        ``(field_name, padding_key)`` tuples.
        """
        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            instance.index_fields(self.vocab)
            padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
            if padding_noise > 0.0:
                noisy_lengths = {}
                for field_name, field_lengths in padding_lengths.items():
                    noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
                padding_lengths = noisy_lengths
            instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                      for (field_name, padding_key) in sorting_keys],
                                     instance)
            instances_with_lengths.append(instance_with_lengths)
        instances_with_lengths.sort(key=lambda x: x[0])
        return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

    @classmethod
    def from_params(cls, params: Params) -> 'BucketIterator':
        sorting_keys = params.pop('sorting_keys')
        padding_noise = params.pop_float('padding_noise', 0.1)
        biggest_batch_first = params.pop_bool('biggest_batch_first', False)
        batch_size = params.pop_int('batch_size', 32)
        instances_per_epoch = params.pop_int('instances_per_epoch', None)
        max_instances_in_memory = params.pop_int('max_instances_in_memory', None)
        params.assert_empty(cls.__name__)
        return cls(sorting_keys=sorting_keys,
                   padding_noise=padding_noise,
                   biggest_batch_first=biggest_batch_first,
                   batch_size=batch_size,
                   instances_per_epoch=instances_per_epoch,
                   max_instances_in_memory=max_instances_in_memory)
