import pytest

from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestSequenceTaggingDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_default_format(self, lazy):
        reader = SequenceTaggingDatasetReader(lazy=lazy)
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "sequence_tagging.tsv")
        instances = ensure_list(instances)

        assert len(instances) == 4
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == ["cats", "are", "animals", "."]
        assert fields["tags"].labels == ["N", "V", "N", "N"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == ["dogs", "are", "animals", "."]
        assert fields["tags"].labels == ["N", "V", "N", "N"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == ["snakes", "are", "animals", "."]
        assert fields["tags"].labels == ["N", "V", "N", "N"]
        fields = instances[3].fields
        assert [t.text for t in fields["tokens"].tokens] == ["birds", "are", "animals", "."]
        assert fields["tags"].labels == ["N", "V", "N", "N"]

    def test_brown_corpus_format(self):
        reader = SequenceTaggingDatasetReader(word_tag_delimiter="/")
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "brown_corpus.txt")
        instances = ensure_list(instances)

        assert len(instances) == 4
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == ["cats", "are", "animals", "."]
        assert fields["tags"].labels == ["N", "V", "N", "N"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == ["dogs", "are", "animals", "."]
        assert fields["tags"].labels == ["N", "V", "N", "N"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == ["snakes", "are", "animals", "."]
        assert fields["tags"].labels == ["N", "V", "N", "N"]
        fields = instances[3].fields
        assert [t.text for t in fields["tokens"].tokens] == ["birds", "are", "animals", "."]
        assert fields["tags"].labels == ["N", "V", "N", "N"]
