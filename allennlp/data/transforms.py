"""
Datasets work really nicely.

The main problem now is how allennlp's batching interacts with the pytorch DataLoader.

At a first glance, this works:

```
def allennlp_collocate(batch):
    batch = AllennlpBatch(batch)
    batch.index_instances(vocab)
    return batch.as_tensor_dict(batch.get_padding_lengths())
```
batch_generator = DataLoader(dataset, batch_size=32, collate_fn=allennlp_collocate)

However, this only works if we want to do very basic batching. In particular,
it can only batch elements which are returned together and because it is a function
passed to `DataLoader`, it also has to be stateless. This is problematic,
because allennlp has several iteration flags which are _not_ stateless.

For example, `maximum_samples_per_batch` takes an existing batch of instances, and
checks the number of _tokens_ present in a particular field. If the max_samples exceeds
the limit, it splits the batch, caching the left over part. This is not possible using
`colocate_fn`.


In order to overcome this problem, I've envisioned something similar to the 
torchvision.Transform api for pre-processing images, but working on the level of entire
datasets.

The idea is that all the steps in the pipeline (indexing, batching, bucketing, filtering etc)
can be written as generators, which can then be wrapped by pytorch datasets.

"""

from typing import Dict, Tuple, List, Iterable, Generic, TypeVar, Deque
import itertools
from collections import deque, defaultdict


import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as IterableTorchDataset


from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

from allennlp.data.fields import MetadataField

from allennlp.common.util import lazy_groups_of


Batched = Iterable[Instance]
InstanceOrBatch = TypeVar("InstanceOrBatch", Iterable[Instance], Instance)


class DatasetFromList(TorchDataset):
    def __init__(self, instances: Iterable[InstanceOrBatch]):
        self.instances

    def __getitem__(self, idx) -> InstanceOrBatch:

        return self.instances[idx]


class DatasetFromGenerator(IterableTorchDataset):
    def __init__(self, generator: Iterable[InstanceOrBatch]):
        self.generator = generator

    def __iter__(self) -> InstanceOrBatch:

        for x in self.generator:
            yield x


class Transform(IterableTorchDataset, Generic[InstanceOrBatch], Registrable):
    def transform(self, dataset: Iterable[Instance]) -> Iterable[InstanceOrBatch]:
        """
        Describes a transformation from either:

        Instance -> Instance (e.g inplace mutation, indexing)
        Instance -> Iterable[Instance] (batching, reading X number of instances into memory)
        """

        raise NotImplementedError

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
                # IMPORTANT! These have to be yield from. because some
                # transforms themeselves return something that is iterable.
                yield from self.transform(example)

                for example in iterable:
                    yield from self.transform(example)

        return DatasetFromGenerator(generator())


@Transform.register("max_instances_in_memory")
class MaxInstancesInMemory(Transform[Batched]):
    """
    turns a dataset into a dataset of chunks of size max_instances_in_memory.
    This is helpful if you have an IterableDataset which you want to read a chunk from
    so you can sort it by padding, and then batch afterward.
    """

    def __init__(self, max_instances_in_memory: int):
        self.max_instances_in_memory = max_instances_in_memory

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        batch = []

        for instance in dataset:
            batch.append(instance)

            if len(batch) == self.max_instances_in_memory:
                yield batch

                batch = []
        if batch:
            yield batch


@Transform.register("batch")
class Batch(Transform[Batched]):
    """
    Batches a dataset.
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


@Transform.register("index")
class Index(Transform[Instance]):
    """
    Indexes allennlp Instances in place and returns them.
    """

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:

        for instance in dataset:
            instance.index_fields(self.vocab)

            yield instance


@Transform.register("sort_by_padding")
class SortByPadding(Transform[Batched]):
    def __init__(self, sorting_keys: List[Tuple[str, str]], padding_noise: float = 0.1):

        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        # HACK, just so we can use the existing sort_by_padding,
        # only works if instances are indexed already.
        self.vocab = None

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        instances = list(dataset)
        if not all([i.indexed for i in instances]):
            raise ValueError("Index() must occur before SortByPadding()")

        # TODO(Mark): Move this to somewhere where it is importable at the top level
        from allennlp.data.iterators.bucket_iterator import sort_by_padding as allennlp_sort_by_padding
        instances = allennlp_sort_by_padding(
            instances, self.sorting_keys, self.vocab, self.padding_noise
        )

        yield instances


@Transform.register("epoch_tracker")
class EpochTracker(Transform[Instance]):
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
class SkipSmallerThan(Transform[Batched]):

    # TODO(Mark): this could collect smaller sub batches
    # and yield them once it has enough, maybe.

    def __init__(self, min_size: int):
        self.min_size = min_size

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        batch = list(dataset)
        if len(batch) >= self.min_size:
            yield batch


# This could also be generic over Iterable[Instance], Iterable[Batched]....
@Transform.register("stop_after")
class StopAfter(Transform[Instance]):
    def __init__(self, max: int):
        self.max = max

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:
        i = 0
        for instance in dataset:
            yield instance
            i += 1
            if i >= self.max:
                break


@Transform.register("max_samples_per_batch")
class MaxSamplesPerBatch(Transform[Batched]):
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
class HomogenousBatchesOf(Transform[Batched]):
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


@Transform.register("fork")
class Fork(Transform[Instance]):
    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:

        info = torch.utils.data.get_worker_info()
        i = 0
        for instance in dataset:

            if info is None:
                yield instance
            elif i % info.num_workers == info.id:
                yield instance
            i += 1


@Transform.register("compose")
class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, dataset: Iterable[Instance]) -> Iterable[InstanceOrBatch]:

        for t in self.transforms:
            dataset = t(dataset)

        yield from dataset

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
