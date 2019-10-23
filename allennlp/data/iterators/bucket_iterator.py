import logging
from typing import List, Tuple, cast, Dict


from allennlp.common.util import add_noise_to_dict_values
from allennlp.data import transforms
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.transform_iterator import TransformIterator
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def sort_by_padding(
    instances: List[Instance],
    sorting_keys: List[Tuple[str, str]],
    vocab: Vocabulary,
    padding_noise: float = 0.0,
) -> List[Instance]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for instance in instances:
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths = noisy_lengths
        instance_with_lengths = (
            [
                padding_lengths[field_name][padding_key]
                for (field_name, padding_key) in sorting_keys
            ],
            instance,
        )
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]


@DataIterator.register("bucket")
class BucketIterator(DataIterator):
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
    maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
        See :class:`BasicIterator`.
    skip_smaller_batches : bool, optional, (default = False)
        When the number of data samples is not dividable by `batch_size`,
        some batches might be smaller than `batch_size`.
        If set to `True`, those smaller batches will be discarded.
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
    ) -> None:

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

        dataset_transforms.append(transforms.Batch(batch_size))

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
