from typing import Dict, Tuple, List, Iterable, Generic, TypeVar, Deque, Union, Any
import itertools
from collections import deque, defaultdict
import random

import torch
from torch.utils.data import IterableDataset as IterableTorchDataset


from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

from allennlp.data.fields import MetadataField

from allennlp.common.util import lazy_groups_of


Batched = Iterable[Instance]
InstanceOrBatch = TypeVar("InstanceOrBatch", Iterable[Instance], Instance)


class DatasetFromGenerator(IterableTorchDataset):
    def __init__(self, generator: Iterable[Union[Iterable[Instance], Instance]]):
        self.generator = generator

    def __iter__(self) -> Iterable[Union[Iterable[Instance], Instance]]:

        for x in self.generator:
            yield x


class Transform(IterableTorchDataset, Registrable):
    """
    A completely generic implementation of a dataset tranformation.
    The Dataset can be an iterable of anything, producing an iterable of
    anything else. This is completely generic so people can implement their
    own transforms which might not fit our strict Instance -> Batch implementation
    in InstanceTransform below. An example of this might be a Transform which
    takes a pytorch Dataset of paths to a sharded dataset and produces a stream
    of instances by reading data from disk.
    """

    def transform(self, dataset: Iterable[Any]) -> Iterable[Any]:
        """
        Takes an Iterable of A, typically a pytorch dataset and transforms it
        into an Iterable of something else.
        """
        raise NotImplementedError

    def transform_batch(self, batches: Iterable[Any]) -> Iterable[Any]:

        raise NotImplementedError

    def __call__(self, dataset: Iterable[Any]) -> Iterable[Any]:

        raise NotImplementedError


class InstanceTransform(Transform, Generic[InstanceOrBatch]):
    def transform(self, dataset: Iterable[Instance]) -> Iterable[InstanceOrBatch]:
        """
        Describes a transformation from either:

        Instance -> Instance (e.g inplace mutation, indexing)
        Instance -> Iterable[Instance] (batching, reading X number of instances into memory)
        """

        raise NotImplementedError

    def transform_batch(self, batches: Iterable[Batched]) -> Iterable[Batched]:

        # TODO Maybe switch this so that the base class just yields and the
        # subclasses which are generic over Batched yield from
        for batch in batches:
            yield from self.transform(batch)  # type: ignore

    def __call__(self, dataset: Iterable[Instance]) -> Iterable[InstanceOrBatch]:
        # wrapper to make sure transform only has to be
        # Iterable[Instance] -> Iterable[InstanceOrBatch],
        # and we handle dispatching the transform based on what type the dataset
        # passed to call is iterable over.

        def generator():
            # Here, we want to 'peek' at the generator
            # to see if it is nested or not.

            iterable = iter(dataset)
            example = next(iterable)
            if isinstance(example, Instance):
                yield from self.transform(itertools.chain([example], iterable))
            else:
                yield from self.transform_batch(itertools.chain([example], iterable))

        return DatasetFromGenerator(generator())


@Transform.register("batch")
class Batch(InstanceTransform[Batched]):
    """
    Batches a dataset. Note that this is exactly the same
    as MaxInstancesInMemory, but we have a duplicated class
    with different parameter names for a nice API.

    Parameters
    ----------
    batch_size: int
        The batch size to convert instances into.
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:
        batch = []

        for instance in dataset:
            batch.append(instance)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


@Transform.register("max_instances_in_memory")
class MaxInstancesInMemory(Batch):
    """
    Turns a dataset into a dataset of chunks of size max_instances_in_memory.
    This is helpful if you have an IterableDataset which you want to read a chunk from
    so you can sort it by padding, and then batch afterward.

    Parameters
    ----------
    max_instances_in_memory : int
        The size of the chunk to read into memory.

    """

    def __init__(self, max_instances_in_memory: int):
        super().__init__(batch_size=max_instances_in_memory)


@Transform.register("index")
class Index(InstanceTransform[Instance]):
    """
    Indexes allennlp Instances in place and returns them.
    """

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:

        for instance in dataset:
            instance.index_fields(self.vocab)

            yield instance

    def transform_batch(self, batches: Iterable[Batched]) -> Iterable[Batched]:

        for batch in batches:
            # Note this is not yield from
            yield self.transform(batch)


@Transform.register("sort_by_padding")
class SortByPadding(InstanceTransform[Batched]):

    """
    Sorts a list of instances by padding and returns them.

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
    """

    def __init__(self, sorting_keys: List[Tuple[str, str]], padding_noise: float = 0.1):

        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        instances = list(dataset)
        if not all([i.indexed for i in instances]):
            raise ValueError("Index() must occur before SortByPadding()")

        # TODO(Mark): Move this to somewhere where it is importable at the top level
        from allennlp.data.iterators.bucket_iterator import (
            sort_by_padding as allennlp_sort_by_padding,
        )

        instances = allennlp_sort_by_padding(instances, self.sorting_keys, None, self.padding_noise)

        yield instances


@Transform.register("epoch_tracker")
class EpochTracker(InstanceTransform[Instance]):
    """
    Adds a allennlp Field to each Instance which specifies how many
    times the full dataset has been iterated over.
    """

    def __init__(self):
        self.epoch = 0

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:

        for instance in dataset:
            instance.fields["epoch_num"] = MetadataField(self.epoch)
            yield instance
        self.epoch += 1


@Transform.register("skip_smaller_than")
class SkipSmallerThan(InstanceTransform[Batched]):
    """
    Skip batches that are smaller than a specified size.
    Useful if you don't want the uneven tail of a dataset
    to be returned as a smaller batch.

    Parameters
    ----------
    min_size : int
        The minimum size of a batch. Only batches of size greater
        than or equal to this will be returned.
    """

    def __init__(self, min_size: int):
        self.min_size = min_size

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        raise NotImplementedError(
            "SkipSmallerThan cannot be called on Instances. Please add Batch to your pipeline first."
        )

    def transform_batch(self, batches: Iterable[Batched]) -> Iterable[Batched]:
        for batch in batches:

            # Explicitly create a list from the iterator
            # before we check it's length, so we don't
            # exhaust it.
            batch = list(batch)
            if len(batch) >= self.min_size:
                yield batch


@Transform.register("stop_after")
class StopAfter(InstanceTransform[Instance]):
    """
    Stop an epoch after a fixed number of individual instances.
    NOTE: The TransformIterator will ensure that the next epoch
    starts from where we stopped.

    Parameters
    ----------
    max : int
        The number of instances to yield in a given epoch.
    """

    def __init__(self, max: int):
        self.max = max

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:
        i = 0
        for instance in dataset:
            yield instance
            i += 1
            if i >= self.max:
                break

    def transform_batch(self, batches: Iterable[Batched]) -> Iterable[Batched]:
        i = 0
        for batch in batches:
            yield batch
            i += 1
            if i >= self.max:
                break


@Transform.register("max_samples_per_batch")
class MaxSamplesPerBatch(InstanceTransform[Batched]):
    """
    Ensures that batches are smaller than a specified number of tokens
    for a particular paddding key. This is an effective method to control
    the expected memory usage of a model.

    Parameters
    ----------
    maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
        If specified, then is a tuple (padding_key, limit) and we will ensure
        that every batch is such that batch_size * sequence_length <= limit
        where sequence_length is given by the padding_key. This is done by
        moving excess instances to the next batch (as opposed to dividing a
        large batch evenly) and should result in a fairly tight packing.
    """

    def __init__(self, max_samples: Tuple[str, int]):

        self.max_samples = max_samples
        # TODO(Mark): Think about whether we want the excess to be across "datasets",
        # as in many cases this will be batches.
        self.excess: Deque[Instance] = deque()

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        instances = list(dataset)

        if not all([i.indexed for i in instances]):
            raise ValueError("Index() must occur before MaxSamplesPerBatch()")

        yield from self._ensure_batch_is_sufficiently_small(instances, self.excess)

    def _ensure_batch_is_sufficiently_small(
        self, batch_instances: List[Instance], excess: Deque[Instance]
    ) -> Iterable[Batched]:
        """
        If self._maximum_samples_per_batch is specified, then split the batch
        into smaller sub-batches if it exceeds the maximum size.

        Parameters
        ----------
        batch_instances : ``Iterable[Instance]``
            A candidate batch.
        excess : ``Deque[Instance]``
            Instances that were not sufficient to form an entire batch
            previously. They will be used as part of the first sub-batch. This
            will be populated with instances from the end of batch_instances
            that do not consist of more than self._maximum_samples_per_batch
            samples or self._batch_size instances. It is the caller's
            responsibility to place these in a batch too, which may, of course,
            be done in part with subsequent calls to this method.

            WARNING: Mutated in place!
        """
        key, limit = self.max_samples

        batch: List[Instance] = []
        padding_length = -1
        original_batch_size = len(batch_instances)

        excess.extend(batch_instances)
        while excess:
            instance = excess.popleft()

            field_lengths = instance.get_padding_lengths()
            for _, lengths in field_lengths.items():
                try:
                    padding_length = max(padding_length, lengths[key])
                except KeyError:
                    pass

            proposed_batch_size = len(batch) + 1
            # Adding the current instance would exceed the batch size or sample size.
            if batch and (
                proposed_batch_size >= original_batch_size
                or padding_length * proposed_batch_size > limit
            ):
                # Output the already existing batch
                yield batch

                # Put the current instance back, reset state.
                excess.appendleft(instance)
                batch = []
                padding_length = -1
            else:
                batch.append(instance)

        if batch:
            yield batch


@Transform.register("homogenous_batches_of")
class HomogenousBatchesOf(InstanceTransform[Batched]):

    """
    This Transform takes a dataset of potentially heterogeneous instances
    and yields back homogeneous batches. It assumes that each instance has
    some ``MetadataField`` indicating what "type" of instance it is
    and bases its notion of homogeneity on that (and, in particular, not on
    inspecting the "field signature" of the instance.)

    Parameters
    ----------
    batch_size : ``int``, required
        The batch size to use to batch instances.
    partition_key : ``str``, optional, (default = "dataset")
        The key of the ``MetadataField`` indicating what "type" of instance this is.
    in_metadata : ``boool``: optional (default = False)
        Whether the partition key actually exists in a Field with the name "metadata",
        rather than being a field itself.
    """

    def __init__(self, batch_size: int, partition_key: str = "dataset", in_metadata: bool = False):

        self.batch_size = batch_size
        self.partition_key = partition_key
        self.in_metadata = in_metadata

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        instances = list(dataset)

        hoppers: Dict[str, List[Instance]] = defaultdict(list)

        for instance in instances:
            if self.in_metadata:
                partition = instance.fields["metadata"].metadata[self.partition_key]  # type: ignore
            else:
                partition = instance.fields[self.partition_key].metadata  # type: ignore
            hoppers[partition].append(instance)

        # Get a `lazy_groups_of` iterator over each set of homogeneous instances.
        batches = {
            key: lazy_groups_of(iter(hopper), self.batch_size) for key, hopper in hoppers.items()
        }

        remaining = set(batches)
        # Yield batches in a round-robin fashion until none are left.
        while remaining:
            for key, lazy_batches in batches.items():
                if key in remaining:
                    try:
                        batch = next(lazy_batches)
                        yield batch
                    except StopIteration:
                        remaining.remove(key)


@Transform.register("biggest_batch_first")
class BiggestBatchFirst(InstanceTransform[Batched]):
    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        raise NotImplementedError(
            "BiggestBatchFirst must be called on batches. "
            "Add a transform which batches your instances to your pipeline."
        )

    def transform_batch(self, batches: Iterable[Batched]) -> Iterable[Batched]:

        batches = list(batches)
        if len(batches) <= 2:
            yield from batches
        else:
            last = batches.pop()
            penultimate = batches.pop()
            yield from [last, penultimate] + batches


@Transform.register("shuffle")
class Shuffle(InstanceTransform[Batched]):
    """
    Shuffles a set of instances and returns them.

    NOTE: This could be generic over batches and instances, i.e
    for the batch case it just shuffles the batches. This is purposely NOT
    like that, because it makes it easy to accidentally shoot yourself
    in the foot if you have a dataset which is infinite, because in the batched
    case, we still try to read all batches into memory. This way,
    if this is called on a dataset which is already batched, it shuffles within
    the batch and returns it lazily.
    """

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        dataset = list(dataset)
        random.shuffle(dataset)
        yield dataset


@Transform.register("fork")
class Fork(InstanceTransform[Instance]):
    """
    A transform which forks a dataset being read by multiple workers
    into independent streams. Each worker specified by the DataLoader
    will read instances modulo it's worker id.
    """

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:

        info = torch.utils.data.get_worker_info()
        i = 0
        for instance in dataset:

            if info is None:
                yield instance
            elif i % info.num_workers == info.id:
                yield instance
            i += 1

    def transform_batch(self, batches: Iterable[Batched]) -> Iterable[Batched]:

        info = torch.utils.data.get_worker_info()
        i = 0
        for batch in batches:
            if info is None:
                yield batch
            elif i % info.num_workers == info.id:
                yield batch
            i += 1


@Transform.register("compose")
class Compose(InstanceTransform):
    """
    A Transform which composes a list of other Transforms.
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def transform(self, dataset: Iterable[InstanceOrBatch]) -> Iterable[InstanceOrBatch]:

        for t in self.transforms:
            dataset = t(dataset)  # type: ignore
        yield from dataset

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
