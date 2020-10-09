import logging
import random
from typing import List, Iterable, Iterator, TypeVar, Sequence

from allennlp.data.instance import Instance
from allennlp.data.samplers.batch_sampler import BatchSampler
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler


logger = logging.getLogger(__name__)


A = TypeVar("A")


@BatchSampler.register("max_tokens_sampler")
class MaxTokensBatchSampler(BucketBatchSampler):
    """
    An sampler which by default, argsorts batches with respect to the maximum input lengths `per
    batch`. Batches are then created such that the number of tokens in a batch does not exceed the given
    maximum number of tokens. You can provide a list of field names and padding keys (or pass none, in which case
    they will be inferred) which the dataset will be sorted by before doing this batching, causing inputs
    with similar length to be batched together, making computation more efficient (as less time is
    wasted on padded elements of the batch).

    # Parameters

    max_tokens : `int`
        The maximum number of tokens to include in a batch.

    sorting_keys : `List[str]`, optional
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        Specifying the right keys for this is a bit cryptic, so if this is not given we try to
        auto-detect the right keys by iterating through a few instances upfront, reading all of the
        padding keys and seeing which one has the longest length.  We use that one for padding.
        This should give reasonable results in most cases. Some cases where it might not be the
        right thing to do are when you have a `ListField[TextField]`, or when you have a really
        long, constant length `TensorField`.

        When you need to specify this yourself, you can create an instance from your dataset and
        call `Instance.get_padding_lengths()` to see a list of all keys used in your data.  You
        should give one or more of those as the sorting keys here.

    padding_noise : `float`, optional (default = `0.1`)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    """

    def __init__(
        self,
        max_tokens: int,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1,
    ):
        super().__init__(-1, sorting_keys, padding_noise, False)
        self.max_tokens = max_tokens

    def _lazy_groups_of_max_size(
        self,
        iterable: Iterable[A],
        sizes: Iterable[int],
    ) -> Iterator[List[A]]:
        """
        Takes an `iterable` of data and an iterable `sizes` of the same length which represents the sizes of each
        corresponding item in `iterable`. The instances from `iterable` are batched such that the total size
        of the batch as computed from `sizes` does not exceed `max_size`.
        """
        cur_max_size = 0
        group: List[A] = []

        iterator = iter(iterable)
        size_iter = iter(sizes)

        for item, size in zip(iterator, size_iter):
            if size > self.max_tokens:
                logger.warning(
                    "Found instance of size %d, which is bigger than the expected size for a batch (%d)",
                    size,
                    self.max_tokens,
                )
            group_size = max(size, cur_max_size) * (len(group) + 1)

            if group_size > self.max_tokens:
                yield group
                cur_max_size = 0
                group = []

            group.append(item)
            cur_max_size = max(cur_max_size, size)

        if len(group) != 0:
            yield group

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        indices, lengths = self._argsort_by_padding(instances)

        max_lengths = [max(length) for length in lengths]
        group_iterator = self._lazy_groups_of_max_size(indices, max_lengths)

        batches = [list(group) for group in group_iterator]
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        # There is no easy way to count the number of batches, so we need to iterate and count.
        return sum(1 for _ in self.get_batch_indices(instances))
