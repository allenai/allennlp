import torch

from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField
from allennlp.data.dataset_readers import MultiTaskDatasetReader
from allennlp.data.data_loaders.multitask_data_loader import MultiTaskDataLoader
from allennlp.data.data_loaders.multitask_scheduler import RoundRobinScheduler
from allennlp.data.data_loaders.multitask_epoch_sampler import UniformSampler


class FakeDatasetReaderA(DatasetReader):
    def _read(self, file_path: str):
        while True:
            yield Instance({"label": LabelField("A")})


class FakeDatasetReaderB(DatasetReader):
    def _read(self, file_path: str):
        while True:
            yield Instance({"label": LabelField("B")})


class MultiTaskDataLoaderTest:
    def test_loading(self):
        reader = MultiTaskDatasetReader(
            readers={"a": FakeDatasetReaderA(), "b": FakeDatasetReaderB()}
        )
        data_path = {"a": "ignored", "b": "ignored"}
        batch_size = 4
        scheduler = RoundRobinScheduler()
        sampler = UniformSampler()
        instances_per_epoch = 8
        batch_size_multiplier = {"a": 1, "b": 2}
        loader = MultiTaskDataLoader(
            reader=reader,
            data_path=data_path,
            batch_size=batch_size,
            scheduler=scheduler,
            sampler=sampler,
            instances_per_epoch=instances_per_epoch,
            batch_size_multiplier=batch_size_multiplier,
            max_instances_in_memory={"a": 10, "b": 10},
        )
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(["A", "B"], "labels")
        loader.index_with(vocab)
        iterator = iter(loader)
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([0, 1, 0]))
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([1, 0]))
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([1, 0]))
        batch = next(iterator)
        assert torch.all(batch["label"] == torch.IntTensor([1]))
