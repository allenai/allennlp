from collections import defaultdict, deque
from typing import List, Tuple, Iterable, Deque
import logging
import random

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.transform_iterator import TransformIterator
from allennlp.data import transforms


logger = logging.getLogger(__name__)


def split_by_language(instance_list):
    insts_by_lang = defaultdict(lambda: [])
    for inst in instance_list:
        inst_lang = inst.fields["metadata"].metadata["lang"]
        insts_by_lang[inst_lang].append(inst)

    return iter(insts_by_lang.values())


@DataIterator.register("same_language")
class SameLanguageIteratorStub:
    """
    Splits batches into batches containing the same language.
    The language of each instance is determined by looking at the 'lang' value
    in the metadata.

    It takes the same parameters as :class:`allennlp.data.iterators.BucketIterator`
    """

    def __new__(
        cls,
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
    ):

        dataset_transforms: List[transforms.Transform] = []

        if instances_per_epoch:
            dataset_transforms.append(transforms.StopAfter(instances_per_epoch))

        if track_epoch:
            dataset_transforms.append(transforms.EpochTracker())

        if max_instances_in_memory is not None:
            dataset_transforms.append(transforms.MaxInstancesInMemory(max_instances_in_memory))

        if sorting_keys is not None:
            # To sort the dataset, it must be indexed.
            # currently this happens when we call index_with, slightly odd
            dataset_transforms.append(transforms.SortByPadding(sorting_keys, padding_noise))

        if maximum_samples_per_batch is not None:
            dataset_transforms.append(transforms.MaxSamplesPerBatch(maximum_samples_per_batch))

        dataset_transforms.append(
            transforms.HomogenousBatchesOf(
                batch_size=batch_size, partition_key="lang", in_metadata=True
            )
        )
        if skip_smaller_batches:
            dataset_transforms.append(transforms.SkipSmallerThan(batch_size))

        return TransformIterator(dataset_transforms, instances_per_epoch, batch_size)

    # TODO(Mark): Explain in detail this delinquent behaviour
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

        pass


@DataIterator.register("same_language_old")
class SameLanguageIterator(BucketIterator):
    """
    Splits batches into batches containing the same language.
    The language of each instance is determined by looking at the 'lang' value
    in the metadata.
    It takes the same parameters as :class:`allennlp.data.iterators.BucketIterator`
    """

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)
            instance_list = split_by_language(instance_list)
            for same_lang_batch in instance_list:
                iterator = iter(same_lang_batch)
                excess: Deque[Instance] = deque()
                # Then break each memory-sized list into batches.
                for batch_instances in lazy_groups_of(iterator, self._batch_size):
                    for poss_smaller_batches in self._ensure_batch_is_sufficiently_small(
                        batch_instances,  # type: ignore
                        excess,
                    ):
                        batch = Batch(poss_smaller_batches)
                        yield batch
                if excess:
                    yield Batch(excess)
