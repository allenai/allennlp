from typing import List, Iterable, Tuple, Dict, cast
import logging
from torch.utils import data

from allennlp.common.registrable import Registrable

from allennlp.common.util import add_noise_to_dict_values, lazy_groups_of
from allennlp.common.lazy import Lazy
from allennlp.data.batch import Batch as AllennlpBatch
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Token
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

logger = logging.getLogger(__name__)


class Sampler(Registrable):
    def __iter__(self) -> Iterable[int]:

        raise NotImplementedError


class BatchSampler(Registrable):
    def __iter__(self) -> Iterable[List[int]]:

        raise NotImplementedError


@Sampler.register("sequential")
class SequentialSampler(Sampler, data.SequentialSampler):
    def __init__(self, data_source: data.Dataset, **kwargs):
        super().__init__(data_source)


@Sampler.register("random")
class RandomSampler(Sampler, data.RandomSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(
        self,
        data_source: data.Dataset,
        replacement: bool = False,
        num_samples: int = None,
        **kwargs,
    ):
        super().__init__(data_source, replacement, num_samples)


@Sampler.register("subset_random")
class SubsetRandomSampler(Sampler, data.SubsetRandomSampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices: List[int], **kwargs):
        super().__init__(indices)


@Sampler.register("weighted_random")
class WeightedRandomSampler(Sampler, data.WeightedRandomSampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights: List[float], num_samples: int, replacement: bool = True, **kwargs):
        super().__init__(weights, num_samples, replacement)


@BatchSampler.register("basic")
class BasicBatchSampler(BatchSampler, data.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool, **kwargs):
        super().__init__(sampler, batch_size, drop_last)


@BatchSampler.register("bucket")
class BatchInstanceSampler(BatchSampler):
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

    def _argsort_by_padding(self, instances: List[Instance]) -> List[int]:
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
        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return [instance_with_index[-1] for instance_with_index in with_indices]

    def __iter__(self) -> Iterable[List[int]]:

        indices = self._argsort_by_padding(self.data_source)
        for group in lazy_groups_of(indices, self._batch_size):
            yield list(group)

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

    def __len__(self):
        return len(self.data_source) // self._batch_size


def allennlp_collocate(batch):
    batch = AllennlpBatch(batch)
    return batch.as_tensor_dict(batch.get_padding_lengths())

from allennlp.common.from_params import FromParams

class DataLoader(FromParams):

    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Sampler = None,
        batch_sampler: BatchSampler = None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn=None,
        multiprocessing_context: str = None,
    ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partially_constructed_sampler = sampler
        self.partially_constructed_batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

    def construct(self, dataset: data.Dataset) -> data.DataLoader:

        collate_fn = allennlp_collocate
        if self.partially_constructed_batch_sampler is not None:
            batch_sampler = self.partially_constructed_batch_sampler.construct(data_source=dataset)
        else:
            batch_sampler = None
        if self.partially_constructed_sampler is not None:
            sampler = self.partially_constructed_sampler.construct(data_source=dataset)
        else:
            sampler = None

        data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            multiprocessing_context=self.multiprocessing_context,
        )
