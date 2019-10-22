
import logging
from typing import Dict, Union, Iterable, Iterator, List, Tuple
from collections import defaultdict
import itertools
import math

import torch
from torch.utils.data import DataLoader


from allennlp.common.util import is_lazy, ensure_list
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import transforms

logger = logging.getLogger(__name__)

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


@DataIterator.register("transform")
class TransformIterator(DataIterator):

    def __init__(
        self,
        sorting_keys: List[Tuple[str, str]] = None,
        padding_noise: float = 0.1,
        biggest_batch_first: bool = False,
        batch_size: int = 32,
        instances_per_epoch: int = None,
        max_instances_in_memory: int = None,
        cache_instances: bool = False, # TODO(Mark): Depreciate argument
        track_epoch: bool = False,
        maximum_samples_per_batch: Tuple[str, int] = None,
        skip_smaller_batches: bool = None
    ) -> None:

        self.vocab: Vocabulary = None

        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        self._skip_smaller_batches = skip_smaller_batches

        self._batch_size = batch_size
        self._max_instances_in_memory = max_instances_in_memory
        self._instances_per_epoch = instances_per_epoch
        self._maximum_samples_per_batch = maximum_samples_per_batch

        # We also might want to add the epoch number to each instance.
        self._track_epoch = track_epoch
        self._epochs: Dict[int, int] = defaultdict(int)

        # We also might want to keep track of cursors;
        # for example, if each epoch represents less than one pass through the dataset,
        # we want to remember where we left off. As `Iterator`s are not necessarily hashable,
        # we use their id() as the key.
        self._cursors: Dict[int, Iterator[Instance]] = {}

        dataset_transforms: List[transforms.Transform] = []

        # BE CAREFUL, mustnt Fork twice. Remember to check once transforms
        # can be passed via constructor.
        dataset_transforms.append(
            transforms.Fork()
        )

        if instances_per_epoch:
            dataset_transforms.append(
                transforms.StopAfter(instances_per_epoch)
            )

        if track_epoch:
            dataset_transforms.append(
                transforms.EpochTracker()
            )

        if max_instances_in_memory is not None:
            dataset_transforms.append(
                transforms.MaxInstancesInMemory(max_instances_in_memory)
            )

        if sorting_keys is not None:
            # To sort the dataset, it must be indexed.
            # currently this happens when we call index_with, slightly odd
            dataset_transforms.append(
                transforms.SortByPadding(sorting_keys, padding_noise)
            )

        if maximum_samples_per_batch is not None:
            dataset_transforms.append(
                transforms.MaxSamplesPerBatch(maximum_samples_per_batch)
            )

        dataset_transforms.append(
            transforms.Batch(batch_size)
        )

        if skip_smaller_batches:
            dataset_transforms.append(
                transforms.SkipSmallerThan(batch_size)
            )

        self.transforms = dataset_transforms

    def __call__(
        self, instances: Iterable[Instance], num_epochs: int = None, shuffle: bool = True
    ) -> Iterator[TensorDict]:

        # Instances is likely to be a list, which cannot be used as a key,
        # so we take the object id instead.
        key = id(instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            for batch in self._create_batches(instances, shuffle):

                yield batch.as_tensor_dict(batch.get_padding_lengths())

    def _collocate(self, batch: Batch) -> TensorDict:

        # If we've added a Batch() into the pipeline,
        # this is a length one list containing a batch.
        # So we unpack it.
        if len(batch) == 1:
            batch = list(batch[0])
        batch = Batch(batch)

        # We might have already done this - but it doesn't matter if we have,
        # because if so it's a no-op.
        batch.index_instances(self.vocab)
        return batch

    def index_with(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        self.transforms = [transforms.Index(vocab)] + self.transforms

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        This method should return one epoch worth of batches.
        """
        max_instances = self._instances_per_epoch

        if max_instances is None:
            data = transforms.Compose(self.transforms)(instances)
            batch_generator = DataLoader(data, batch_size=1, collate_fn=self._collocate)
            yield from batch_generator

        else:
            # If we don't have a cursor for this dataset, create one. We use ``id()``
            # for the key because ``instances`` could be a list, which can't be used as a key.
            key = id(instances)

            iterator = self._cursors.get(key, itertools.cycle(instances))
            data = transforms.Compose(self.transforms)(iterator)
            batch_generator = iter(DataLoader(data, batch_size=1, collate_fn=self._collocate))

            while max_instances > 0:
                try:
                    # If there are instances left on this iterator,
                    # yield one and decrement max_instances.
                    next_batch = next(batch_generator)
                    yield next_batch
                    max_instances -= len(next_batch.instances)

                # TODO explain this because it's hella confusing
                except StopIteration:
                    break

            # We may have a new iterator, so update the cursor.
            self._cursors[key] = iterator

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
