import logging
from typing import Dict, Union, Iterable, Iterator, List, Optional, Tuple
from collections import defaultdict
import itertools
import math
import random

import torch

from allennlp.common.registrable import Registrable
from allennlp.common.util import is_lazy, lazy_groups_of, ensure_list
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]  # pylint: disable=invalid-name


def add_epoch_number(batch: Batch, epoch: int) -> Batch:
    """
    Add the epoch number to the batch instances as a MetadataField.
    """
    for instance in batch.instances:
        instance.fields['epoch_num'] = MetadataField(epoch)
    return batch


class DataIterator(Registrable):
    """
    An abstract ``DataIterator`` class. ``DataIterators`` must override ``_create_batches()``.

    Parameters
    ----------
    batch_size : ``int``, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : ``int``, optional, (default = None)
        If specified, each epoch will consist of precisely this many instances.
        If not specified, each epoch will consist of a single pass through the dataset.
    max_instances_in_memory : ``int``, optional, (default = None)
        If specified, the iterator will load this many instances at a time into an
        in-memory list and then produce batches from one such list at a time. This
        could be useful if your instances are read lazily from disk.
    cache_instances : ``bool``, optional, (default = False)
        If true, the iterator will cache the tensorized instances in memory.
        If false, it will do the tensorization anew each iteration.
    track_epoch : ``bool``, optional, (default = False)
        If true, each instance will get a ``MetadataField`` containing the epoch number.
    maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
        If specified, then is a tuple (padding_key, limit) and we will
        shrink the batch size for very long sequences such that
        batch_size * sequence_length <= limit where sequence_length is given
        by the padding_key.
    """
    default_implementation = 'bucket'

    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        self.vocab: Vocabulary = None

        self._batch_size = batch_size
        self._max_instances_in_memory = max_instances_in_memory
        self._instances_per_epoch = instances_per_epoch
        self._maximum_samples_per_batch = maximum_samples_per_batch

        # We might want to cache the instances in memory.
        self._cache_instances = cache_instances
        self._cache: Dict[int, List[TensorDict]] = defaultdict(list)

        # We also might want to add the epoch number to each instance.
        self._track_epoch = track_epoch
        self._epochs: Dict[int, int] = defaultdict(int)

        # We also might want to keep track of cursors;
        # for example, if each epoch represents less than one pass through the dataset,
        # we want to remember where we left off. As `Iterator`s are not necessarily hashable,
        # we use their id() as the key.
        self._cursors: Dict[int, Iterator[Instance]] = {}


    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True,
                 cuda_device: int = -1) -> Iterator[TensorDict]:
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.

        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset. IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        num_epochs : ``int``, optional (default=``None``)
            How times should we iterate over this dataset?  If ``None``, we will iterate over it
            forever.
        shuffle : ``bool``, optional (default=``True``)
            If ``True``, we will shuffle the instances in ``dataset`` before constructing batches
            and iterating over the data.
        cuda_device : ``int``
            If cuda_device >= 0, GPUs are available and Pytorch was compiled with CUDA support, the
            tensor will be copied to the cuda_device specified.
        """
        # Instances is likely to be a list, which cannot be used as a key,
        # so we take the object id instead.
        key = id(instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            self._epochs[key] = epoch

            if self._cache_instances and key in self._cache:
                # Serve the results from the cache.
                tensor_dicts = self._cache[key]

                if shuffle:
                    random.shuffle(tensor_dicts)
                for tensor_dict in tensor_dicts:
                    if self._track_epoch:
                        # The tensor_dict already has an "epoch_num" tensor,
                        # so just fill it with the right value.
                        epoch_tensor: torch.Tensor = tensor_dict['epoch_num']
                        epoch_tensor.fill_(epoch)
                    yield tensor_dict
            else:
                batches = self._create_batches(instances, shuffle)

                # Should we add the instances to the cache this epoch?
                add_to_cache = self._cache_instances and key not in self._cache

                for batch in batches:
                    if self._track_epoch:
                        add_epoch_number(batch, epoch)

                    if self.vocab is not None:
                        batch.index_instances(self.vocab)

                    padding_lengths = batch.get_padding_lengths()
                    logger.debug("Batch padding lengths: %s", str(padding_lengths))
                    logger.debug("Batch size: %d", len(batch.instances))
                    tensor_dict = batch.as_tensor_dict(padding_lengths,
                                                       cuda_device=cuda_device)

                    if add_to_cache:
                        self._cache[key].append(tensor_dict)

                    yield tensor_dict

    def _take_instances(self,
                        instances: Iterable[Instance],
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
            iterator = self._cursors.get(key, iter(instances))

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
            self._cursors[key] = iterator

    def _memory_sized_lists(self,
                            instances: Iterable[Instance]) -> Iterable[List[Instance]]:
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
        iterator = self._take_instances(instances, self._instances_per_epoch)

        # We have four different cases to deal with:

        # With lazy instances and no guidance about how many to load into memory,
        # we just load ``batch_size`` instances at a time:
        if lazy and self._max_instances_in_memory is None:
            yield from lazy_groups_of(iterator, self._batch_size)
        # If we specified max instances in memory, lazy or not, we just
        # load ``max_instances_in_memory`` instances at a time:
        elif self._max_instances_in_memory is not None:
            yield from lazy_groups_of(iterator, self._max_instances_in_memory)
        # If we have non-lazy instances, and we want all instances each epoch,
        # then we just yield back the list of instances:
        elif self._instances_per_epoch is None:
            yield ensure_list(instances)
        # In the final case we have non-lazy instances, we want a specific number
        # of instances each epoch, and we didn't specify how to many instances to load
        # into memory. So we convert the whole iterator to a list:
        else:
            yield list(iterator)


    def _ensure_batch_is_sufficiently_small(self, batch_instances: Iterable[Instance]) -> List[List[Instance]]:
        """
        If self._maximum_samples_per_batch is specified, then split the batch into smaller
        sub-batches if it exceeds the maximum size.
        """
        if self._maximum_samples_per_batch is None:
            return [list(batch_instances)]

        # check if we need to break into smaller chunks
        key, limit = self._maximum_samples_per_batch
        padding_length = -1
        list_batch_instances = list(batch_instances)
        for instance in list_batch_instances:
            field_lengths = instance.get_padding_lengths()
            for _, lengths in field_lengths.items():
                try:
                    padding_length = max(padding_length,
                                         lengths[key])
                except KeyError:
                    pass

        if padding_length * len(list_batch_instances) > limit:
            # need to shrink
            num_samples = padding_length * len(list_batch_instances)
            num_shrunk_batches = math.ceil(num_samples / float(limit))
            shrunk_batch_size = math.ceil(len(list_batch_instances) / num_shrunk_batches)
            shrunk_batches = []
            start = 0
            while start < len(list_batch_instances):
                end = start + shrunk_batch_size
                shrunk_batches.append(list_batch_instances[start:end])
                start = end
            return shrunk_batches
        else:
            return [list_batch_instances]


    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        """
        Returns the number of batches that ``dataset`` will be split into; if you want to track
        progress through the batch with the generator produced by ``__call__``, this could be
        useful.
        """
        if is_lazy(instances) and self._instances_per_epoch is None:
            # Unable to compute num batches, so just return 1.
            return 1
        elif self._instances_per_epoch is not None:
            return math.ceil(self._instances_per_epoch / self._batch_size)
        else:
            # Not lazy, so can compute the list length.
            return math.ceil(len(ensure_list(instances)) / self._batch_size)


    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        This method should return one epoch worth of batches.
        """
        raise NotImplementedError

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab
