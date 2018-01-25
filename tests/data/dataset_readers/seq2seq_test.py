# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import Seq2SeqDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestSeq2SeqDatasetReader(AllenNlpTestCase):
    def test_default_format(self):
        reader = Seq2SeqDatasetReader()
        generator = reader.instance_generator('tests/fixtures/data/seq2seq_copy.tsv')
        instances = list(generator())

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
        generator = reader.instance_generator('tests/fixtures/data/seq2seq_copy.tsv')
        instances = list(generator())

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["this", "is", "a", "sentence", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "a", "sentence", "@@END@@"]
