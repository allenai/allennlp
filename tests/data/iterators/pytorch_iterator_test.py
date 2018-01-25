# pylint: disable=no-self-use,invalid-name
from typing import List, Callable, Iterator

import torch.utils.data as data

from allennlp.data import Instance
from allennlp.data.dataset import Dataset as Batch
from tests.data.iterators.basic_iterator_test import IteratorTest

class Dataset(data.Dataset):
    """
    Eager dataset
    """
    def __init__(self, instances: List[Instance]) -> None:
        super().__init__()
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Instance:
        return self.instances[idx]


class LazyDataset(data.Dataset):
    """
    This gets a list of *generators* (say, one per file)
    """
    def __init__(self, generators: List[Callable[[], Iterator[Instance]]]) -> None:
        super().__init__()
        self.generators = generators

    def __len__(self) -> int:
        return len(self.generators)

    def __getitem__(self, idx: int) -> Iterator[Instance]:
        return self.generators[idx]()


def collate(batch: List[Instance]) -> Batch:
    """
    Collate instances from a non-lazy dataset
    """
    return Batch(batch)

def lazy_collate(batch: List[Iterator[Instance]]) -> Batch:
    """
    Collate instances from a lazy dataset,
    where we're actually being handed a batch of generators
    """
    instances = [instance for iterator in batch for instance in iterator]
    return Batch(instances)

def tokens(instance: Instance) -> List[str]:
    """
    Pull the tokens out of the `text` field (as strings).
    This is just so we can compare identical instances
    that may not be identical python objects
    (e.g. if they're loaded in a multiprocessing way)
    """
    return [token.text for token in instance.fields['text'].tokens]

def same_instances(instances1: List[Instance], instances2: List[Instance], same_order: bool = True) -> bool:
    """
    Check if the two lists of instances are the same, by just pulling out
    the tokens from the `text` field and comparing them. Order may or may not matter.
    """
    tokens1 = [tokens(i1) for i1 in instances1]
    tokens2 = [tokens(i2) for i2 in instances2]

    if not same_order:
        tokens1.sort()
        tokens2.sort()
    return tokens1 == tokens2


class TestPytorchIterator(IteratorTest):
    def setUp(self):
        super().setUp()
        self.dataset = Dataset(self.instances)
        # Lazy dataset has 10 lazy copies of the instances
        self.lazy_dataset = LazyDataset([lambda: iter(self.instances) for _ in range(10)])

    def test_loader(self):
        # Important to specify `collate` function
        # that turns Instances into a Batch.
        # By default each batch size is 1.
        loader = data.DataLoader(self.dataset, collate_fn=collate)
        batches = [batch for batch in loader]
        assert len(batches) == len(self.instances)
        instances = [instance for batch in batches for instance in batch]
        assert same_instances(instances, self.instances)

    def test_loader_with_batch_size(self):
        # Try again with batch size 2.
        loader = data.DataLoader(self.dataset, collate_fn=collate, batch_size=2)
        batches = [batch for batch in loader]
        assert len(batches) == 3
        sizes = [len(batch.instances) for batch in batches]
        assert sizes == [2, 2, 1]
        instances = [instance for batch in batches for instance in batch]
        assert same_instances(instances, self.instances)

    def test_loader_with_shuffle(self):
        # With 100 x 5 instances, there's a good probability that
        # shuffling will produce a different order.
        dataset = Dataset(self.instances * 100)
        loader = data.DataLoader(dataset, collate_fn=collate, shuffle=True)
        batches = [batch for batch in loader]
        instances = [instance for batch in batches for instance in batch]
        assert instances != self.instances * 100
        assert same_instances(instances, self.instances * 100, same_order=False)

    def test_lazy_loader(self):
        # Lazy loader produces one batch per *generator*.
        loader = data.DataLoader(self.lazy_dataset, collate_fn=lazy_collate, num_workers=2)
        batches = [batch for batch in loader]
        assert len(batches) == 10
        instances = [instance for batch in batches for instance in batch]
        assert same_instances(instances, self.instances * 10, same_order=False)

    def test_lazy_loader_with_batch_size(self):
        # Note: this batch_size of 2 means two *iterators* at a time
        loader = data.DataLoader(self.lazy_dataset, collate_fn=lazy_collate, batch_size=2, num_workers=0)
        batches = [batch for batch in loader]
        # 10 iterators / 2 at a time => 5 batches
        assert len(batches) == 5
        instances = [instance for batch in batches for instance in batch]
        assert len(instances) == 50  # 5 * 10
        assert same_instances(instances, self.instances * 10, same_order=False)
