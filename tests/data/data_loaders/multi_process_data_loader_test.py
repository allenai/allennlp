import pytest

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.data_loaders.multi_process_data_loader import MultiProcessDataLoader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer


def test_error_raised_when_text_fields_contain_token_indexers():
    """
    This tests that the MultiProcessDataLoader raises an error when num_workers > 0
    but the dataset reader doesn't implement apply_token_indexers().
    """

    class BadDatasetReader(DatasetReader):
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

    with pytest.raises(ValueError, match="Make sure your dataset reader's text_to_instance()"):
        loader = MultiProcessDataLoader(
            BadDatasetReader(), "this-path-doesn't-matter", num_workers=2
        )
        list(loader.iter_instances())


class MockDatasetReader(DatasetReader):
    """
    We'll use this mock dataset reader for most of the tests.

    It utilizes a transformers tokenizer, since historically we've deadlocking
    issues when using these within subprocesses. So these tests also serve as
    regression tests against those issues.
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
            yield self.text_to_instance(source, target)

    def text_to_instance(self, source: str, target: str = None) -> Instance:  # type: ignore
        fields = {}
        fields["source"] = TextField(self.tokenizer.tokenize(source))
        if target is not None:
            fields["target"] = TextField(self.tokenizer.tokenize(target))
        return Instance(fields)  # type: ignore

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source"].token_indexers = self.token_indexers  # type: ignore
        if "target" in instance.fields:
            instance.fields["target"].token_indexers = self.token_indexers  # type: ignore
