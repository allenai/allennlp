from typing import List, Iterable
from torch.utils import data

from allennlp.common.registrable import Registrable

"""
Duplicates of the pytorch Sampler classes. Broadly, these only exist
so that we can add type hints, meaning we can construct them from configuration
files. You can use these directly from Python code, but they are identical to the
pytorch ones.
"""


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
class SequentialSampler(data.SequentialSampler, Sampler):
    """
    A registrable version of pytorch's
    [SequentialSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.SequentialSampler).

    Registered as a `Sampler` with name "sequential".

    In a typical AllenNLP configuration file, `data_source` parameter does not get an entry under
    the "sampler", it gets constructed separately.
    """

    def __init__(self, data_source: data.Dataset):
        super().__init__(data_source)


@Sampler.register("random")
class RandomSampler(data.RandomSampler, Sampler):
    """
    A registrable version of pytorch's
    [RandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler).
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify `num_samples` to draw.

    Registered as a `Sampler` with name "random".

    # Parameters
    data_source: `Dataset`, required
        The dataset to sample from.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "sampler", it gets constructed separately.
    replacement : `bool`, optional (default = `False`)
        Samples are drawn with replacement if `True`.
    num_samples: `int` (default = `len(dataset)`)
        The number of samples to draw. This argument
        is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(
        self, data_source: data.Dataset, replacement: bool = False, num_samples: int = None
    ):
        super().__init__(data_source, replacement, num_samples)


@Sampler.register("subset_random")
class SubsetRandomSampler(data.SubsetRandomSampler, Sampler):
    """
    A registrable version of pytorch's
    [SubsetRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler).
    Samples elements randomly from a given list of indices, without replacement.

    Registered as a `Sampler` with name "subset_random".

    # Parameters
    indices: `List[int]`
        a sequence of indices to sample from.
    """

    def __init__(self, indices: List[int]):
        super().__init__(indices)


@Sampler.register("weighted_random")
class WeightedRandomSampler(data.WeightedRandomSampler, Sampler):
    """
    A registrable version of pytorch's
    [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler).
    Samples elements from `[0,...,len(weights)-1]` with given probabilities (weights).

    Registered as a `Sampler` with name "weighted_random".

    # Parameters:
    weights : `List[float]`
        A sequence of weights, not necessary summing up to one.
    num_samples : `int`
        The number of samples to draw.
    replacement : `bool`
        If ``True``, samples are drawn with replacement.
        If not, they are drawn without replacement, which means that when a
        sample index is drawn for a row, it cannot be drawn again for that row.

    # Examples

    ```python
    >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
    [0, 0, 0, 1, 0]
    >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
    [0, 1, 4, 3, 2]
    ```
    """

    def __init__(self, weights: List[float], num_samples: int, replacement: bool = True):
        super().__init__(weights, num_samples, replacement)


@BatchSampler.register("basic")
class BasicBatchSampler(data.BatchSampler, BatchSampler):
    """
    A registrable version of pytorch's
    [BatchSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler).
    Wraps another sampler to yield a mini-batch of indices.

    Registered as a `BatchSampler` with name "basic".

    # Parameters
    sampler: `Sampler`
        The base sampler.
    batch_size : `int`
        The size of the batch.
    drop_last : `bool`
        If `True`, the sampler will drop the last batch if
        its size would be less than batch_size`.

    # Examples

    ```python
    >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    ```
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)
