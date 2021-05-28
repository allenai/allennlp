from typing import List, Iterable, Dict

import torch
import pytest
from allennlp.common.testing import requires_gpu
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.data_loaders import MultiProcessDataLoader, WorkerError
from allennlp.data.fields import Field, TextField, MetadataField, TensorField
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.data_loaders.data_collator import LanguageModelingDataCollator


class MockDatasetReader(DatasetReader):
    """
    We'll use this mock dataset reader for most of the tests.

    It utilizes a transformers tokenizer, since historically we've seen deadlocking
    issues when using these within subprocesses. So these tests also serve as
    regression tests against those issues.

    And unlike `MockOldDatasetReader` below, it implements all of the new API,
    specifically the `apply_token_indexers` method, so that it can be used
    with num_workers > 0.
    """

    NUM_INSTANCES = 100

    def __init__(self, model: str = "epwalsh/bert-xsmall-dummy", **kwargs) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self.tokenizer = PretrainedTransformerTokenizer(model)
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(model)}

    def _read(self, file_path: str):
        for i in self.shard_iterable(range(self.NUM_INSTANCES)):
            source = f"Hi there, I'm the {i}th instance"
            target = f"Hello, {i}th instance!"
            yield self.text_to_instance(i, source, target)

    def text_to_instance(self, index: int, source: str, target: str = None) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}
        fields["source"] = TextField(self.tokenizer.tokenize(source))
        fields["index"] = MetadataField(index)  # type: ignore
        # It's important to have tests that use a tensor field since sending tensors
        # between processes has a lot of pitfalls.
        fields["tensor"] = TensorField(torch.tensor([1, 2, 3]))
        if target is not None:
            fields["target"] = TextField(self.tokenizer.tokenize(target))
        return Instance(fields)  # type: ignore

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source"].token_indexers = self.token_indexers  # type: ignore
        if "target" in instance.fields:
            instance.fields["target"].token_indexers = self.token_indexers  # type: ignore


class MockOldDatasetReader(DatasetReader):
    def __init__(self, model: str = "epwalsh/bert-xsmall-dummy", **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = PretrainedTransformerTokenizer(model)
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(model)}

    def _read(self, file_path: str):
        for i in range(10):
            source = f"Hi there, I'm the {i}th instance"
            target = f"Hello, {i}th instance!"
            yield self.text_to_instance(source, target)

    def text_to_instance(self, source: str, target: str = None) -> Instance:  # type: ignore
        fields = {}
        fields["source"] = TextField(self.tokenizer.tokenize(source), self.token_indexers)  # type: ignore
        if target is not None:
            fields["target"] = TextField(self.tokenizer.tokenize(target), self.token_indexers)  # type: ignore
        return Instance(fields)  # type: ignore


@pytest.mark.parametrize("max_instances_in_memory", (None, 10))
def test_error_raised_when_text_fields_contain_token_indexers(max_instances_in_memory):
    """
    This tests that the MultiProcessDataLoader raises an error when num_workers > 0
    but the dataset reader doesn't implement apply_token_indexers().

    It also tests that errors raised within a worker process are propogated upwards
    to the main process, and that when that happens, all workers will be successfully
    killed.
    """

    with pytest.raises(WorkerError, match="Make sure your dataset reader's text_to_instance()"):
        loader = MultiProcessDataLoader(
            MockOldDatasetReader(),
            "this-path-doesn't-matter",
            num_workers=2,
            max_instances_in_memory=max_instances_in_memory,
            batch_size=1,
        )
        list(loader.iter_instances())


@pytest.mark.parametrize(
    "options",
    [
        dict(max_instances_in_memory=10, num_workers=2, batch_size=1),
        dict(num_workers=2, batch_size=1),
        dict(max_instances_in_memory=10, num_workers=2, start_method="spawn", batch_size=1),
        dict(num_workers=2, start_method="spawn", batch_size=1),
        dict(max_instances_in_memory=10, num_workers=0, batch_size=1),
        dict(num_workers=0, batch_size=1),
    ],
    ids=str,
)
def test_multiprocess_data_loader(options):
    reader = MockDatasetReader()
    data_path = "this doesn't matter"

    loader = MultiProcessDataLoader(reader=reader, data_path=data_path, **options)
    if not options.get("max_instances_in_memory"):
        # Instances should be loaded immediately if max_instances_in_memory is None.
        assert loader._instances

    instances: Iterable[Instance] = loader.iter_instances()
    # This should be a generator.
    assert not isinstance(instances, (list, tuple))
    instances = list(instances)
    assert len(instances) == MockDatasetReader.NUM_INSTANCES

    # Now build vocab.
    vocab = Vocabulary.from_instances(instances)

    # Before indexing the loader, trying to iterate through batches will raise an error.
    with pytest.raises(ValueError, match="Did you forget to call DataLoader.index_with"):
        list(loader)

    loader.index_with(vocab)

    # Run through a couple epochs to make sure we collect all of the instances.
    for epoch in range(2):
        indices: List[int] = []
        for batch in loader:
            for index in batch["index"]:
                indices.append(index)  # type: ignore
        # Ensure no duplicates.
        assert len(indices) == len(set(indices)), indices
        # Ensure all collected.
        assert len(indices) == MockDatasetReader.NUM_INSTANCES, epoch


def test_drop_last():
    """
    Ensures that the `drop_last` option is respected.
    """
    loader = MultiProcessDataLoader(MockDatasetReader(), "some path", batch_size=16, drop_last=True)
    vocab = Vocabulary.from_instances(loader.iter_instances())
    loader.index_with(vocab)

    # Should still load all instances. `drop_last` only affects batches.
    assert len(list(loader.iter_instances())) == MockDatasetReader.NUM_INSTANCES

    # Just here because the assertions below depend on the exact value of NUM_INSTANCES.
    assert MockDatasetReader.NUM_INSTANCES == 100
    batches = list(loader)
    for batch in batches:
        assert len(batch["index"]) == 16
    assert len(batches) == 6


def test_language_model_data_collator():
    """
    Ensure `LanguageModelingDataCollator` works
    """
    norm_loader = MultiProcessDataLoader(MockDatasetReader(), "some path", batch_size=16)
    vocab = Vocabulary.from_instances(norm_loader.iter_instances())
    norm_loader.index_with(vocab)
    batch0 = list(norm_loader)[0]

    model_name = "epwalsh/bert-xsmall-dummy"
    data_collate = LanguageModelingDataCollator(model_name)
    mlm_loader = MultiProcessDataLoader(
        MockDatasetReader(), "some path", batch_size=16, collate_fn=data_collate
    )
    vocab = Vocabulary.from_instances(mlm_loader.iter_instances())
    mlm_loader.index_with(vocab)
    batch1 = list(mlm_loader)[0]

    norm_inputs = batch0["source"]["tokens"]["token_ids"]
    mlm_inputs = batch1["source"]["tokens"]["token_ids"]
    mlm_labels = batch1["source"]["tokens"]["labels"]

    # if we replace the mlm inputs with their labels, should be same as origin inputs
    assert torch.where(mlm_labels != -100, mlm_labels, mlm_inputs).tolist() == norm_inputs.tolist()


def test_batches_per_epoch():
    loader = MultiProcessDataLoader(
        MockDatasetReader(), "some path", batch_size=4, batches_per_epoch=10
    )
    vocab = Vocabulary.from_instances(loader.iter_instances())
    loader.index_with(vocab)

    assert len(loader) == 10
    assert len(list(loader)) == 10


@pytest.mark.parametrize(
    "options",
    [
        dict(num_workers=0, batch_size=2),
        dict(num_workers=1, batch_size=2),
        dict(num_workers=1, batch_size=2, start_method="spawn"),
    ],
    ids=str,
)
@requires_gpu
def test_load_to_cuda(options):
    reader = MockDatasetReader()
    loader = MultiProcessDataLoader(
        reader=reader,
        data_path="this doens't matter",
        cuda_device=0,
        **options,
    )
    vocab = Vocabulary.from_instances(loader.iter_instances())
    loader.index_with(vocab)
    for batch in loader:
        assert batch["tensor"].device == torch.device("cuda:0")
