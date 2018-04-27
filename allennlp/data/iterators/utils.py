"""
Utilities for creating DataIterators.
"""
from typing import Iterable, Optional, List, Iterator, Dict, cast, Tuple

from allennlp.common.util import lazy_groups_of, ensure_list, is_lazy, add_noise_to_dict_values
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

# Global variable for tracking cursors.
_cursors: Dict[int, Iterator[Instance]] = {}

def take_instances(instances: Iterable[Instance],
                   max_instances: Optional[int] = None) -> Iterator[Instance]:
    """
    Take the next `max_instances` instances from the given dataset.
    If `max_instances` is `None`, then just take all instances from the dataset.
    If `max_instances` is not `None`, each call resumes where the previous one
    left off, and when you get to the end of the dataset you start again from the beginning.
    """
    # If max_instances isn't specified, just iterate once over the whole dataset
    if max_instances is None:
        yield from iter(instances)
    else:
        # If we don't have a cursor for this dataset, create one. We use ``id()``
        # for the key because ``instances`` could be a list, which can't be used as a key.
        key = id(instances)
        iterator = _cursors.get(key, iter(instances))

        while max_instances > 0:
            try:
                # If there are instances left on this iterator,
                # yield one and decrement max_instances.
                yield next(iterator)
                max_instances -= 1
            except StopIteration:
                # None left, so start over again at the beginning of the dataset.
                iterator = iter(instances)

        # We may have a new iterator, so update the cursor.
        _cursors[key] = iterator


def memory_sized_lists(instances: Iterable[Instance],
                       batch_size: int,
                       max_instances_in_memory: Optional[int] = None,
                       instances_per_epoch: Optional[int] = None) -> Iterable[List[Instance]]:
    """
    Breaks the dataset into "memory-sized" lists of instances,
    which it yields up one at a time until it gets through a full epoch.

    For example, if the dataset is already an in-memory list, and each epoch
    represents one pass through the dataset, it just yields back the dataset.
    Whereas if the dataset is lazily read from disk and we've specified to
    load 1000 instances at a time, then it yields lists of 1000 instances each.
    """
    lazy = is_lazy(instances)

    # Get an iterator over the next epoch worth of instances.
    iterator = take_instances(instances, instances_per_epoch)

    # We have four different cases to deal with:

    # With lazy instances and no guidance about how many to load into memory,
    # we just load ``batch_size`` instances at a time:
    if lazy and max_instances_in_memory is None:
        yield from lazy_groups_of(iterator, batch_size)
    # If we specified max instances in memory, lazy or not, we just
    # load ``max_instances_in_memory`` instances at a time:
    elif max_instances_in_memory is not None:
        yield from lazy_groups_of(iterator, max_instances_in_memory)
    # If we have non-lazy instances, and we want all instances each epoch,
    # then we just yield back the list of instances:
    elif instances_per_epoch is None:
        yield ensure_list(instances)
    # In the final case we have non-lazy instances, we want a specific number
    # of instances each epoch, and we didn't specify how to many instances to load
    # into memory. So we convert the whole iterator to a list:
    else:
        yield list(iterator)

def add_epoch_number(batch: Batch, epoch: int) -> Batch:
    """
    Add the epoch number to the batch instances as a MetadataField.
    """
    for instance in batch.instances:
        instance.fields['epoch_num'] = MetadataField(epoch)
    return batch

def sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary,
                    padding_noise: float = 0.0) -> List[Instance]:
    """
    Sorts the ``Instances`` in this ``Batch`` by their padding lengths, using the keys in
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
        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]
