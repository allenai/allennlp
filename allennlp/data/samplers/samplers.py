from typing import List, Iterable, Tuple, Dict, cast
import logging
from torch.utils import data

from allennlp.common.registrable import Registrable

from allennlp.common.util import add_noise_to_dict_values, lazy_groups_of
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)


class Sampler(Registrable):
    """
    A copy of the pytorch [Sampler](https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html)
    which allows us to register it with `Registrable.`
    """

    def __iter__(self) -> Iterable[int]:

        raise NotImplementedError


class BatchSampler(Registrable):
    """
    A copy of the pytorch
    [BatchSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler)
    which allows us to register it with `Registrable.`
    """

    def __iter__(self) -> Iterable[List[int]]:

        raise NotImplementedError


@Sampler.register("sequential")
class SequentialSampler(Sampler, data.SequentialSampler):
    """
    A registerable version of pytorch's
    [SequentialSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.SequentialSampler).
    """

    def __init__(self, data_source: data.Dataset):
        super().__init__(data_source)


@Sampler.register("random")
class RandomSampler(Sampler, data.RandomSampler):
    """
    A registerable version of pytorch's
    [RandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler).
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify `num_samples` to draw.

    # Parameters
    data_source: `Dataset`, reqired
        The dataset to sample from.
    replacement : `bool`, optional(default = False)
        Samples are drawn with replacement if `True`.
    num_samples: `int` (default = `len(dataset)`)
        The number of samples to draw. This argument
        is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(
        self, data_source: data.Dataset, replacement: bool = False, num_samples: int = None,
    ):
        super().__init__(data_source, replacement, num_samples)


@Sampler.register("subset_random")
class SubsetRandomSampler(Sampler, data.SubsetRandomSampler):
    """
    A registerable version of pytorch's
    [SubsetRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler).
    Samples elements randomly from a given list of indices, without replacement.

    # Parameters
    indices: `List[int]`
        a sequence of indices to sample from.
    """

    def __init__(self, indices: List[int]):
        super().__init__(indices)


@Sampler.register("weighted_random")
class WeightedRandomSampler(Sampler, data.WeightedRandomSampler):
    """
    A registerable version of pytorch's
    [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler).
    Samples elements from `[0,...,len(weights)-1]` with given probabilities (weights).

    # Parameters:
    weights : `List[float]`
        A sequence of weights, not necessary summing up to one.
    num_samples : `int`
        The number of samples to draw.
    replacement : `bool`
        If ``True``, samples are drawn with replacement.
        If not, they are drawn without replacement, which means that when a
        sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
    ```
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    ```
    """

    def __init__(self, weights: List[float], num_samples: int, replacement: bool = True):
        super().__init__(weights, num_samples, replacement)


@BatchSampler.register("basic")
class BasicBatchSampler(BatchSampler, data.BatchSampler):
    """
    A registerable version of pytorch's
    [BatchSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler).
    Wraps another sampler to yield a mini-batch of indices.

    # Parameters
    sampler: `Sampler`
        The base sampler.
    batch_size : `int`
        The size of the batch.
    drop_last : `bool`
        If `True`, the sampler will drop the last batch if
        its size would be less than batch_size`.

    Example:
    ```
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    ```
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)


@BatchSampler.register("bucket")
class BatchInstanceSampler(BatchSampler):
    """
    An sampler which by default, argsorts batches with respect to the maximum input lengths `per
    batch`. You can provide a list of field names and padding keys (or pass none, in which case they
    will be inferred) which the dataset will be sorted by before doing this batching, causing inputs
    with similar length to be batched together, making computation more efficient (as less time is
    wasted on padded elements of the batch).

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
    batch_size : int, required.
        The size of each batch of instances yielded when calling the dataloader.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.

        Note that if you specify `max_instances_in_memory`, the first batch will only be the
        biggest from among the first "max instances in memory" instances.

    """

    def __init__(
        self,
        data_source: data.Dataset,
        batch_size: int,
        sorting_keys: List[Tuple[str, str]] = None,
        padding_noise: float = 0.1,
    ):

        self.vocab = data_source.vocab
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._batch_size = batch_size
        self.data_source = data_source

    def _argsort_by_padding(self, instances: Iterable[Instance]) -> List[int]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
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
        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return [instance_with_index[-1] for instance_with_index in with_indices]

    def __iter__(self) -> Iterable[List[int]]:

        indices = self._argsort_by_padding(self.data_source)
        for group in lazy_groups_of(indices, self._batch_size):
            yield list(group)

    def _guess_sorting_keys(self, instances: Iterable[Instance], num_instances: int = 10) -> None:
        """
        Use `num_instances` instances from the dataset to infer the keys used
        for sorting the dataset for bucketing.

        # Parameters

        instances : `Iterable[Instance]`, required.
            The dataset to guess sorting keys for.
        num_instances : `int`, optional (default = 10)
            The number of instances to use to guess sorting keys. Typically
            the default value is completely sufficient, but if your instances
            are not homogeneous, you might need more.
        """
        max_length = 0.0
        longest_padding_key: Tuple[str, str] = None
        for i, instance in enumerate(instances):
            instance.index_fields(self.vocab)
            padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
            for field_name, field_padding in padding_lengths.items():
                for padding_key, length in field_padding.items():
                    if length > max_length:
                        max_length = length
                        longest_padding_key = (field_name, padding_key)
            if i > num_instances:
                # Only use num_instances instances to guess the sorting keys.
                break

        if not longest_padding_key:
            # This shouldn't ever happen (you basically have to have an empty instance list), but
            # just in case...
            raise AssertionError(
                "Found no field that needed padding; we are surprised you got this error, please "
                "open an issue on github"
            )
        self._sorting_keys = [longest_padding_key]

    def __len__(self):
        return len(self.data_source) // self._batch_size
