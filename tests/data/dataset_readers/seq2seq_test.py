# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.data.dataset_readers import Seq2SeqDatasetReader
from allennlp.common.util import ensure_list

class TestSeq2SeqDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_default_format(self, lazy):
        reader = Seq2SeqDatasetReader(lazy=lazy)
        instances = reader.read('tests/fixtures/data/seq2seq_copy.tsv')
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "a", "sentence", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "a", "sentence", "@@END@@"]
        fields = instances[1].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "another", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "another", "@@END@@"]
        fields = instances[2].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["@@START@@", "all", "these", "sentences",
                                                                    "should", "get", "copied", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "all", "these", "sentences",
                                                                    "should", "get", "copied", "@@END@@"]

    def test_source_add_start_token(self):
        reader = Seq2SeqDatasetReader(source_add_start_token=False)
        instances = reader.read('tests/fixtures/data/seq2seq_copy.tsv')
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["this", "is", "a", "sentence", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "a", "sentence", "@@END@@"]
