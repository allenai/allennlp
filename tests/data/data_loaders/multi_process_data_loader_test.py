from typing import List, Iterable

import pytest

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.vocabulary import Vocabulary


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

    NUM_INSTANCES = 1000

    def __init__(self, model: str = "epwalsh/bert-xsmall-dummy", **kwargs) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multi_process_sharding=True, **kwargs
        )
        self.tokenizer = PretrainedTransformerTokenizer(model)
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(model)}

    def _read(self, file_path: str):
        for i in self.shard_iterable(range(self.NUM_INSTANCES)):
            source = f"Hi there, I'm the {i}th instance"
            target = f"Hello, {i}th instance!"
            yield self.text_to_instance(i, source, target)

    def text_to_instance(self, index: int, source: str, target: str = None) -> Instance:  # type: ignore
        fields = {}
        fields["source"] = TextField(self.tokenizer.tokenize(source))
        fields["index"] = MetadataField(index)  # type: ignore
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

    with pytest.raises(ValueError, match="Make sure your dataset reader's text_to_instance()"):
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
)
def test_multi_process_data_loader(options):
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
    for _ in range(2):
        indices: List[int] = []
        for batch in loader:
            for index in batch["index"]:
                indices.append(index)  # type: ignore
        assert len(indices) == len(set(indices)) == MockDatasetReader.NUM_INSTANCES
