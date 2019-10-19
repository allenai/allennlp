



from typing import Dict, Tuple, Iterator, List, Callable, Iterable, Union, Generic, TypeVar, Deque
import json
import logging
import itertools
import time
import argparse
from collections import deque, defaultdict

from overrides import overrides

import numpy
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as IterableTorchDataset
from torch.utils.data import DataLoader, Sampler


from allennlp.common.registrable import Registrable
from allennlp.data.dataset import Batch as AllennlpBatch
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

from allennlp.data.dataset_readers import DatasetReader, SnliReader
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from allennlp.data.iterators.bucket_iterator import sort_by_padding as allennlp_sort_by_padding
from allennlp.common.util import lazy_groups_of



"""
Allennlp datasets are now just Pytorch Datasets of allennlp Instances.
These have to implement `text_to_instance`. We have two abstract classes
for each of the indexable and iterable pytorch datasets.

"""

class Dataset(TorchDataset, Registrable):

    def text_to_instance(self, *inputs) -> Instance:
        
        raise NotImplementedError

    def __getitem__(self) -> Instance:

        raise NotImplementedError


class IterableDataset(IterableTorchDataset, Registrable):

    def text_to_instance(self, *inputs) -> Instance:
        
        raise NotImplementedError

    def __iter__(self) -> Iterator[Instance]:

        raise NotImplementedError


"""
Here we have two SNLI readers in both of the different styles.
They are only slightly different.
"""

class SnliDataset(Dataset):

    def __init__(self,
                 file_path: str,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:

        super().__init__()
        
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        file_path = cached_path(file_path)
        self.examples = []

        for line in open(file_path, "r"):
            example = json.loads(line)
            if example["gold_label"] == "-":
                # These were cases where the annotators disagreed; we'll just skip them.  It's
                # like 800 out of 500k examples in the training data.
                continue
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Instance:
        example = self.examples[idx]
        return self.text_to_instance(example["sentence1"], example["sentence2"], example["gold_label"])

    @overrides
    def text_to_instance(self,
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields["premise"] = TextField(premise_tokens, self._token_indexers)
        fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields["label"] = LabelField(label)

        metadata = {
            "premise_tokens": [x.text for x in premise_tokens],
            "hypothesis_tokens": [x.text for x in hypothesis_tokens],
        }
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)


class IterableSnliDataset(IterableDataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:

        super().__init__()
        
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.file_path = cached_path(file_path)

    def __iter__(self) -> Iterator[Instance]:

        for line in open(self.file_path, "r"):

            example = json.loads(line)
            if example["gold_label"] == "-":
                # These were cases where the annotators disagreed; we'll just skip them.  It's
                # like 800 out of 500k examples in the training data.
                continue
            
            yield self.text_to_instance(example["sentence1"], example["sentence2"], example["gold_label"])

    @overrides
    def text_to_instance(self,
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields["premise"] = TextField(premise_tokens, self._token_indexers)
        fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields["label"] = LabelField(label)

        metadata = {
            "premise_tokens": [x.text for x in premise_tokens],
            "hypothesis_tokens": [x.text for x in hypothesis_tokens],
        }
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)


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

Batched = Iterable[Instance]
InstanceOrBatch = TypeVar("InstanceOrBatch", Iterable[Instance], Instance)

class DatasetFromList(TorchDataset):

    def __init__(self, instances: Iterable[InstanceOrBatch]):
        self.instances

    def __getitem__(self, idx) -> InstanceOrBatch:

        return self.instances[i]

class DatasetFromGenerator(IterableTorchDataset):

    def __init__(self, generator: Iterable[InstanceOrBatch]):
        self.generator = generator

    def __iter__(self) -> InstanceOrBatch:

        for x in self.generator:
            yield x


class Transform(IterableTorchDataset, Generic[InstanceOrBatch]):

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
            # Here, we want to 'peek' at the generator to see if it is
            # nested or not. 
            example = next(iter(dataset))
            if isinstance(example, Instance):

                yield from self.transform(itertools.chain([example], dataset))
            else:
                # IMPORTANT! These have to be yield from. because some
                # transforms themeselves return something that is iterable.
                yield from self.transform(example)

                for example in dataset:
                    yield from self.transform(example)

        return DatasetFromGenerator(generator())


class MaxInstancesInMemory(Transform[Batched]):
    """
    turns a dataset into a dataset of chunks of size max_instances_in_memory.
    This is helpful if you have an IterableDataset which you want to read a chunk from
    so you can sort it by padding, and then batch afterward.
    """
    def __init__(self,
        max_instances_in_memory: int
    ):
        self.max_instances_in_memory = max_instances_in_memory

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        batch = []

        for instance in dataset:
            batch.append(instance)

            if len(batch) == self.max_instances_in_memory:
                yield batch
                batch = []

        yield batch


# Batching is actually the same as MaxInstancesInMemory,
# but we also accept this name as conceptually they are thought about differently.
Batch = MaxInstancesInMemory

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
        

class SortByPadding(Transform[Instance]):

    def __init__(self,
        sorting_keys: List[Tuple[str, str]],
        padding_noise: float = 0.1,
        ):
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        # HACK, just so we can use the existing sort_by_padding,
        # only works if instances are indexed already.
        self.vocab = None

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:
        

        instances = list(dataset)
        if not all([i.indexed for i in instances]):
            raise ValueError("Index() must occur before SortByPadding()")

        instances = allennlp_sort_by_padding(
            instances, self.sorting_keys, self.vocab, self.padding_noise)
        
        yield from instances


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
class StopAfter(Transform[Instance]):

    def __init__(self, max: int):
        self.max = max

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Instance]:
        i = 0
        for instance in dataset:
            if i >= self.max:
                break
            yield instance
            i += 1

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

        batches: List[List[Instance]] = []
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

        # Keep the current batch as excess.
        excess.extend(batch)

        return batches


class HomogenousBatchesOf(Transform[Batched]):

    def __init__(self, batch_size: int, partition_key: str = "dataset"):
        
        self.batch_size = batch_size
        self.partition_key = partition_key

    def transform(self, dataset: Iterable[Instance]) -> Iterable[Batched]:

        instances = list(dataset)

        hoppers: Dict[str, List[Instance]] = defaultdict(list)

        for instance in instance_list:
            partition = instance.fields[self.partition_key].metadata  # type: ignore
            hoppers[partition].append(instance)

        # Get a `lazy_groups_of` iterator over each set of homogeneous instances.
        batches = {
            key: lazy_groups_of(iter(hopper), self.batch_size)
            for key, hopper in hoppers.items()
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



class Compose(Transform):

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, dataset: Iterable[Instance]) -> Iterable[InstanceOrBatch]:

        for t in self.transforms:
            dataset = t(dataset)

        yield from dataset

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


"""
Actual demonstration of the API.
"""

data = SnliDataset("snli_20.jsonl")
vocab = Vocabulary.from_instances(data)


def allennlp_collocate(batch):

    # If we've added a Batch() into the pipeline,
    # this is a length one list containing a batch.
    # So we unpack it.
    if len(batch) == 1:
        batch = list(batch[0])

    batch = AllennlpBatch(batch)
    # We might have already done this - but it doesn't matter if we have,
    # because if so it's a no-op. 
    batch.index_instances(vocab)
    return batch.as_tensor_dict(batch.get_padding_lengths())


transformations = [
    Index(vocab),
    SortByPadding([("premise", "num_tokens")]),
    Batch(5),
    MaxSamplesPerBatch(("num_tokens", 100)),
]



data = Compose(transformations)(data)

# Here we pass batch size=1 to the dataloader,
# because we have already done batching in our pipeline e.g collocate_fn 
# is recieving a list of a single batch of instances. 
batch_generator = DataLoader(data, batch_size=1, collate_fn=allennlp_collocate)

print("Example batches")
for batch in batch_generator:
    print(batch)

