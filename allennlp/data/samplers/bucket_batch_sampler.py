import logging
import math
from typing import List, Iterable, Tuple, Sequence, Optional
import random

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers.batch_sampler import BatchSampler


logger = logging.getLogger(__name__)


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise


@BatchSampler.register("bucket")
class BucketBatchSampler(BatchSampler):
    """
    An sampler which by default, argsorts batches with respect to the maximum input lengths `per
    batch`. You can provide a list of field names and padding keys (or pass none, in which case they
    will be inferred) which the dataset will be sorted by before doing this batching, causing inputs
    with similar length to be batched together, making computation more efficient (as less time is
    wasted on padded elements of the batch).

    # Parameters

    batch_size : `int`, required
        The size of each batch of instances yielded when calling the data_loader.

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

    padding_noise : `float`, optional (default=`.1`)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.

    drop_last : `bool`, (default = `False`)
        If `True`, the sampler will drop the last batch if
        its size would be less than batch_size`.

    shuffle : `bool`, (default = `True`)
        If `False`, the sampler won't shuffle the batches. `padding_noise` will be ignored and set
        to `0.0`.

    """

    def __init__(
        self,
        batch_size: int,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if not shuffle:
            self.padding_noise = 0.0

    def _argsort_by_padding(
        self, instances: Iterable[Instance]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
        """
        if not self.sorting_keys:
            logger.info("No sorting keys given; trying to guess a good one")
            self._guess_sorting_keys(instances)
            logger.info(f"Using {self.sorting_keys} as the sorting keys")
        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            lengths = []
            noisy_lengths = []
            for field_name in self.sorting_keys:  # type: ignore
                if field_name not in instance.fields:
                    raise ConfigurationError(
                        f'Sorting key "{field_name}" is not a field in instance. '
                        f"Available fields/keys are {list(instance.fields.keys())}."
                    )
                lengths.append(len(instance.fields[field_name]))

                noisy_lengths.append(add_noise_to_value(lengths[-1], self.padding_noise))
            instances_with_lengths.append((noisy_lengths, lengths, instance))
        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return (
            [instance_with_index[-1] for instance_with_index in with_indices],
            [instance_with_index[0][1] for instance_with_index in with_indices],
        )

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        indices, _ = self._argsort_by_padding(instances)
        batches = []
        for group in lazy_groups_of(indices, self.batch_size):
            batch_indices = list(group)
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batches.append(batch_indices)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def _guess_sorting_keys(self, instances: Iterable[Instance], num_instances: int = 10) -> None:
        """
        Use `num_instances` instances from the dataset to infer the keys used
        for sorting the dataset for bucketing.

        # Parameters

        instances : `Iterable[Instance]`, required.
            The dataset to guess sorting keys for.
        num_instances : `int`, optional (default = `10`)
            The number of instances to use to guess sorting keys. Typically
            the default value is completely sufficient, but if your instances
            are not homogeneous, you might need more.
        """
        max_length = 0.0
        longest_field: Optional[str] = None
        for i, instance in enumerate(instances):
            for field_name, field in instance.fields.items():
                length = len(field)
                if length > max_length:
                    max_length = length
                    longest_field = field_name
            if i > num_instances:
                # Only use num_instances instances to guess the sorting keys.
                break

        if not longest_field:
            # This shouldn't ever happen (you basically have to have an empty instance list), but
            # just in case...
            raise AssertionError(
                "Found no field that needed padding; we are surprised you got this error, please "
                "open an issue on github"
            )
        self.sorting_keys = [longest_field]

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        batch_count_float = len(instances) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)

    def get_batch_size(self) -> Optional[int]:
        return self.batch_size
