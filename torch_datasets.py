



from typing import Dict, Tuple, Iterator, List, Callable
import json
import logging

import time
import argparse

from overrides import overrides

import numpy
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as IterableTorchDataset
from torch.utils.data import DataLoader, Sampler


from allennlp.common.registrable import Registrable
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

from allennlp.data.dataset_readers import DatasetReader, SnliReader
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from allennlp.data.iterators.bucket_iterator import sort_by_padding as allennlp_sort_by_padding
from allennlp.common.util import lazy_groups_of

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


class SnliDataset(Dataset):

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
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
    ) -> Instance:

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
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
    ) -> Instance:

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



class BucketDataset(IterableTorchDataset):

    def __init__(self,
        dataset,
        sorting_keys: List[Tuple[str, str]],
        padding_noise: float = 0.1,
        max_instances_in_memory: int = None
    ):
        super().__init__()

        self.dataset = dataset

        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._max_instances_in_memory = max_instances_in_memory

        self.vocab = None

    def __iter__(self) -> Iterator[List[Instance]]:

        for instance_list in self._memory_sized_lists(self.dataset):
            instance_list = sort_by_padding(
                instance_list, self._sorting_keys, self.vocab, self._padding_noise)
            
            yield instance_list
            #for instance in instance_list:
            #    yield instance

    def _memory_sized_lists(self, instances: Iterator[Instance]) -> Iterator[Iterator[Instance]]: 

        if self._max_instances_in_memory is None:
            yield from [instances]
        elif self._max_instances_in_memory is not None:
            yield from lazy_groups_of(instances, self._max_instances_in_memory)


    def index_with(self, vocab: Vocabulary) -> None:
        self.vocab = vocab



class MaxInstancesInMemory:

    def __init__(self,
        max_instances_in_memory: int
    ):
        self.max_instances_in_memory = max_instances_in_memory

    def __call__(self, dataset: Iterator[Instance]) -> Iterator[List[Instance]]:

        batch = []

        for instance in dataset:
            batch.append(instance)

            if len(batch) == self.max_instances_in_memory:
                yield batch
                batch = []

        yield batch

class Index:

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def __call__(self, dataset: Iterator[Instance]) -> Iterator[Instance]:

        for instance in dataset:
            instance.index_fields(self.vocab)

            yield instance
        

class SortByPadding:

    def __init__(self,
        sorting_keys: List[Tuple[str, str]],
        padding_noise: float = 0.1,
        ):
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.vocab = None # HACK, just so we can use the existing sort_by_padding, only works if instances are indexed already.

    def __call__(self, dataset: Iterator[List[Instance]]) -> Iterator[List[Instance]]:
        
        for instances in dataset:

            if not all([i.indexed for i in instances]):
                raise ValueError("Index() must occur before SortByPadding()")

            instances = allennlp_sort_by_padding(
                instances, self.sorting_keys, self.vocab, self.padding_noise)
            
            yield instances



class EpochTracker:

    def __init__(self):

        self.epoch = 0

    def __call__(self, dataset: Iterator[Instance]) -> Iterator[Instance]:

        for instance in dataset:
            instance.fields["epoch_num"] = MetadataField(self.epoch)
            yield instance
        self.epoch += 1


class Transform(IterableTorchDataset):


    def transform(self, dataset):
        raise NotImplementedError

    def __call__(self, dataset):
        # wrapper to make sure transform only has to be Iterator[Instance] -> Iterator[Union[Instance, List[Instance]]]
        
        def generator():

            example = next(dataset)

            if isinstance(example, list): # !!! check for iterable.
                yield self.transform(example)

                for example in dataset:
                    yield self.transform(example)
            else:
                import itertools
                yield from self.transform(itertools.chain([example], dataset))

        return DatasetFromGenerator(generator())


class DatasetFromList(TorchDataset):

    def __init__(self, instances):
        self.instances

    def __getitem__(self, idx):

        return self.instances[i]

class DatasetFromGenerator(IterableTorchDataset):

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):

        for x in self.generator:
            yield x


class Compose(IterableTorchDataset):

    def __init__(self, dataset, transforms):
        self.transforms = transforms
        for t in transforms:
            dataset = t(dataset)

        self.dataset = dataset

    def __iter__(self) -> Iterator[Instance]:
        for i in self.dataset:
            yield i

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

data = SnliDataset("/Users/markn/allen_ai/allennlp/allennlp/tests/fixtures/data/snli.jsonl")

for x in data:
    print(x)

data_iterable = IterableSnliDataset("/Users/markn/allen_ai/allennlp/allennlp/tests/fixtures/data/snli.jsonl")

for x in data_iterable:
    print(x)

vocab = Vocabulary.from_instances(data)
vocab = Vocabulary.from_instances(data_iterable)



def allennlp_collocate(batch):

    if len(batch) == 1 and isinstance(batch[0], list):
        batch = batch[0]

    batch = Batch(batch)
    # We might have already done this - but it doesn't matter if we have,
    # because if so it's a no-op. 
    batch.index_instances(vocab)
    return batch.as_tensor_dict(batch.get_padding_lengths())


transformations = [
    Index(vocab),
    EpochTracker(),
    MaxInstancesInMemory(1000),
    SortByPadding([("premise", "num_tokens")]),
]


data = Compose(data, transformations)

iterable_dataset = Compose(data_iterable, transformations)


batch_generator = DataLoader(data, batch_size=1, collate_fn=allennlp_collocate)
stream_batch_generator = DataLoader(iterable_dataset, batch_size=1, collate_fn=allennlp_collocate)

print("non iterable")
for batch in batch_generator:
    print(batch)

print("non iterable")
for batch in stream_batch_generator:
    print(batch)

