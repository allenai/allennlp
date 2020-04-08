from typing import Iterable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import DatasetReader, InterleavingDatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer


class PlainTextReader(DatasetReader):
    def __init__(self):
        super().__init__()
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = SpacyTokenizer()

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as input_file:
            for line in input_file:
                yield self.text_to_instance(line)

    def text_to_instance(self, line: str) -> Instance:  # type: ignore

        tokens = self._tokenizer.tokenize(line)
        return Instance({"line": TextField(tokens, self._token_indexers)})


class TestInterleavingDatasetReader(AllenNlpTestCase):
    def test_round_robin(self):
        readers = {"a": PlainTextReader(), "b": PlainTextReader(), "c": PlainTextReader()}

        reader = InterleavingDatasetReader(readers)
        data_dir = self.FIXTURES_ROOT / "data"

        file_path = f"""{{
            "a": "{data_dir / 'babi.txt'}",
            "b": "{data_dir / 'conll2003.txt'}",
            "c": "{data_dir / 'conll2003.txt'}"
        }}"""

        instances = list(reader.read(file_path))
        first_three_keys = {instance.fields["dataset"].metadata for instance in instances[:3]}
        assert first_three_keys == {"a", "b", "c"}

        next_three_keys = {instance.fields["dataset"].metadata for instance in instances[3:6]}
        assert next_three_keys == {"a", "b", "c"}

    def test_all_at_once(self):
        readers = {"f": PlainTextReader(), "g": PlainTextReader(), "h": PlainTextReader()}

        reader = InterleavingDatasetReader(
            readers, dataset_field_name="source", scheme="all_at_once"
        )
        data_dir = self.FIXTURES_ROOT / "data"

        file_path = f"""{{
            "f": "{data_dir / 'babi.txt'}",
            "g": "{data_dir / 'conll2003.txt'}",
            "h": "{data_dir / 'conll2003.txt'}"
        }}"""

        buckets = []
        last_source = None

        # Fill up a bucket until the source changes, then start a new one
        for instance in reader.read(file_path):
            source = instance.fields["source"].metadata
            if source != last_source:
                buckets.append([])
                last_source = source
            buckets[-1].append(instance)

        # should be in 3 buckets
        assert len(buckets) == 3
