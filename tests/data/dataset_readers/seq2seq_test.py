# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import Seq2SeqDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestSeq2SeqDatasetReader(AllenNlpTestCase):
    def test_default_format(self):
        reader = Seq2SeqDatasetReader()
        dataset = reader.read('tests/fixtures/data/seq2seq_copy.tsv')

        assert len(dataset.instances) == 3
        fields = dataset.instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["this", "is", "a", "sentence"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "a", "sentence", "@@END@@"]
        fields = dataset.instances[1].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["this", "is", "another"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "another", "@@END@@"]
        fields = dataset.instances[2].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["all", "these", "sentences", "should", "get",
                                                                    "copied"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "all", "these", "sentences",
                                                                    "should", "get", "copied", "@@END@@"]
