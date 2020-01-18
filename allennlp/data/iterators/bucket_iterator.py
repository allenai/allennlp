import logging
import random
from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque

from overrides import overrides

from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.batch import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator

logger = logging.getLogger(__name__)


@DataIterator.register("bucket")
class BucketIterator(DataIterator):
    """
    An iterator which by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).

    # Parameters

    sorting_keys : List[Tuple[str, str]], optional
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        Specifying the right keys for this is a bit cryptic, so if this is not given we try to
        auto-detect the right keys by iterating once through the data up front, reading all of the
        padding keys and seeing which one has the longest length.  We use that one for padding.
        This should give reasonable results in most cases.

        When you need to specify this yourself, you can create an instance from your dataset and
        call `Instance.get_padding_lengths()` to see a list of all keys used in your data.  You
        should give one or more of those as the sorting keys here.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.

        Note that if you specify `max_instances_in_memory`, the first batch will only be the
        biggest from among the first "max instances in memory" instances.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        See :class:`BasicIterator`.
    max_instances_in_memory : int, optional, (default = None)
        See :class:`BasicIterator`.
    maximum_samples_per_batch : `Tuple[str, int]`, (default = None)
        See :class:`BasicIterator`.
    skip_smaller_batches : bool, optional, (default = False)
        When the number of data samples is not dividable by `batch_size`,
        some batches might be smaller than `batch_size`.
        If set to `True`, those smaller batches will be discarded.
    """

    def __init__(
        self,
        sorting_keys: List[Tuple[str, str]] = None,
        padding_noise: float = 0.1,
        biggest_batch_first: bool = False,
        batch_size: int = 32,
        instances_per_epoch: int = None,
        max_instances_in_memory: int = None,
        cache_instances: bool = False,
        track_epoch: bool = False,
        maximum_samples_per_batch: Tuple[str, int] = None,
        skip_smaller_batches: bool = False,
    ) -> None:
        super().__init__(
            cache_instances=cache_instances,
            track_epoch=track_epoch,
            batch_size=batch_size,
            instances_per_epoch=instances_per_epoch,
            max_instances_in_memory=max_instances_in_memory,
            maximum_samples_per_batch=maximum_samples_per_batch,
        )
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        self._skip_smaller_batches = skip_smaller_batches

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):

            instance_list = self._sort_by_padding(instance_list)

            batches = []
            excess: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(
                    batch_instances, excess
                ):
                    if (
                        self._skip_smaller_batches
                        and len(possibly_smaller_batches) < self._batch_size
                    ):
                        continue
                    batches.append(Batch(possibly_smaller_batches))
            if excess and (not self._skip_smaller_batches or len(excess) == self._batch_size):
                batches.append(Batch(excess))

            # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
            # num_gpu batches together, shuffle and then expand the groups.
            # This guards against imbalanced batches across GPUs.
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)

            yield from batches

    def _sort_by_padding(self, instances: List[Instance]) -> List[Instance]:
        """
        Sorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided).  `sorting_keys` is a list of
        `(field_name, padding_key)` tuples.
        """
        if not self._sorting_keys:
            logger.info("No sorting keys given; trying to guess a good one")
            self._guess_sorting_keys(instances)
            logger.info(f"Using {self._sorting_keys} as the sorting keys")
        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            instance.index_fields(self.vocab)
            padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
            if self._padding_noise > 0.0:
                noisy_lengths = {}
                for field_name, field_lengths in padding_lengths.items():
                    noisy_lengths[field_name] = add_noise_to_dict_values(
                        field_lengths, self._padding_noise
                    )
                padding_lengths = noisy_lengths
            instance_with_lengths = (
                [
                    padding_lengths[field_name][padding_key]
                    for (field_name, padding_key) in self._sorting_keys
                ],
                instance,
            )
            instances_with_lengths.append(instance_with_lengths)
        instances_with_lengths.sort(key=lambda x: x[0])
        return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

    def _guess_sorting_keys(self, instances: List[Instance]) -> None:
        max_length = 0.0
        longest_padding_key: Tuple[str, str] = None
        for instance in instances:
            instance.index_fields(self.vocab)
            padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
            for field_name, field_padding in padding_lengths.items():
                for padding_key, length in field_padding.items():
                    if length > max_length:
                        max_length = length
                        longest_padding_key = (field_name, padding_key)
        if not longest_padding_key:
            # This shouldn't ever happen (you basically have to have an empty instance list), but
            # just in case...
            raise AssertionError(
                "Found no field that needed padding; we are surprised you got this error, please "
                "open an issue on github"
            )
        self._sorting_keys = [longest_padding_key]
