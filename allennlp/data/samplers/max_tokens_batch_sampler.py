import logging
import random
from typing import List, Iterable, Optional

from allennlp.common.util import lazy_groups_of_max_size
from allennlp.data.samplers import BatchSampler, BucketBatchSampler
from torch.utils import data

logger = logging.getLogger(__name__)


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

    data_source: `data.Dataset`, required,
        The pytorch `Dataset` of allennlp Instances to bucket.

    max_tokens : `int`, required
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
        long, constant length `ArrayField`.

        When you need to specify this yourself, you can create an instance from your dataset and
        call `Instance.get_padding_lengths()` to see a list of all keys used in your data.  You
        should give one or more of those as the sorting keys here.

    padding_noise : `float`, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    """

    def __init__(
        self,
        data_source: data.Dataset,
        max_tokens: Optional[int] = None,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1
    ):
        super().__init__(data_source, -1, sorting_keys, padding_noise, False)

        self.max_tokens = max_tokens

    def __iter__(self) -> Iterable[List[int]]:
        indices, lengths = self._argsort_by_padding(self.data_source)

        group_iterator = lazy_groups_of_max_size(indices, lengths, self.max_tokens)

        batches = [list(group) for group in group_iterator]
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        # There is no easy way to count the number of batches, so we need to iterate and count.
        return sum(1 for _ in self)
