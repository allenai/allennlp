import itertools

import pytest
import torch

from allennlp.common.util import cycle_iterator_function
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField
from allennlp.data.dataset_readers import MultiTaskDatasetReader
from allennlp.data.data_loaders.multitask_data_loader import MultiTaskDataLoader
from allennlp.data.data_loaders.multitask_scheduler import RoundRobinScheduler
from allennlp.data.data_loaders.multitask_epoch_sampler import UniformSampler, WeightedSampler


class FakeDatasetReaderA(DatasetReader):
    def _read(self, file_path: str):
        return itertools.islice(
            cycle_iterator_function(lambda: [Instance({"label": LabelField("A")})]), 100
        )


class FakeDatasetReaderB(DatasetReader):
    def _read(self, file_path: str):
        return itertools.islice(
            cycle_iterator_function(lambda: [Instance({"label": LabelField("B")})]), 100
        )


class MultiTaskDataLoaderTest:
    def test_loading(self):
        reader = MultiTaskDatasetReader(
            readers={"a": FakeDatasetReaderA(), "b": FakeDatasetReaderB()}
        )
        data_path = {"a": "ignored", "b": "ignored"}
        scheduler = RoundRobinScheduler(batch_size=4)
        sampler = UniformSampler()
        loader = MultiTaskDataLoader(
            reader=reader,
            data_path=data_path,
            scheduler=scheduler,
            sampler=sampler,
            instances_per_epoch=8,
            max_instances_in_memory={"a": 10, "b": 10},
        )
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(["A", "B"], "labels")
        loader.index_with(vocab)
        iterator = iter(loader)
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([0, 1, 0, 1]))
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([0, 1, 0, 1]))
        with pytest.raises(StopIteration):
            next(iterator)

    def test_loading_with_sampler(self):
        reader = MultiTaskDatasetReader(
            readers={"a": FakeDatasetReaderA(), "b": FakeDatasetReaderB()}
        )
        data_path = {"a": "ignored", "b": "ignored"}
        scheduler = RoundRobinScheduler(batch_size=4)
        sampler = WeightedSampler({"a": 1, "b": 2})
        loader = MultiTaskDataLoader(
            reader=reader,
            data_path=data_path,
            scheduler=scheduler,
            sampler=sampler,
            instances_per_epoch=9,
        )
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(["A", "B"], "labels")
        loader.index_with(vocab)
        iterator = iter(loader)
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([0, 1, 0, 1]))
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([0, 1, 1, 1]))
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([1]))
        with pytest.raises(StopIteration):
            next(iterator)
