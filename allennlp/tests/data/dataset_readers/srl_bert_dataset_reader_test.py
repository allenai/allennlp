# pylint: disable=no-self-use,invalid-name

from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers.srl_bert_reader import SrlBertReader
from allennlp.common.testing.test_case import AllenNlpTestCase


class TestBertSrlReader(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        self.reader = SrlBertReader(bert_model_name="bert-base-uncased")

    def test_convert_tags_to_wordpiece_tags(self):
        # pylint: disable=protected-access
        offsets = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        original = ['B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1',
                    'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']
        wordpiece_tags = ['O', 'B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                          'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'O']
        converted = self.reader._convert_tags_to_wordpiece_tags(original, offsets)
        assert converted == wordpiece_tags

        offsets = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        converted = self.reader._convert_tags_to_wordpiece_tags(original, offsets)
        assert converted == ['O', 'B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                             'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']

        offsets = [2, 4, 6]
        original = ["B-ARG", "B-V", "O"]
        converted = self.reader._convert_tags_to_wordpiece_tags(original, offsets)
        assert converted == ['O', 'B-ARG', 'B-V', 'I-V', 'O', 'O', 'O']

        offsets = [3, 4, 6]
        original = ["B-ARG", "I-ARG", "O"]
        converted = self.reader._convert_tags_to_wordpiece_tags(original, offsets)
        assert converted == ['O', 'B-ARG', 'I-ARG', 'I-ARG', 'O', 'O', 'O']
        # pylint: enable=protected-access

    def test_read_from_file(self):
        conll_reader = self.reader
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'conll_2012' / 'subdomain')
        instances = ensure_list(instances)
        fields = instances[0].fields
        tokens = fields["metadata"]["words"]
        assert tokens == ["Mali", "government", "officials", "say", "the", "woman", "'s",
                          "confession", "was", "forced", "."]
        assert fields["verb_indicator"].labels[4] == 1

        assert fields["tags"].labels == ['O', 'B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                                         'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'O']

        fields = instances[1].fields
        tokens = fields["metadata"]["words"]
        assert tokens == ["Mali", "government", "officials", "say", "the", "woman", "'s",
                          "confession", "was", "forced", "."]
        assert fields["verb_indicator"].labels[10] == 1
        assert fields["tags"].labels == ['O', 'O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1', 'I-ARG1',
                                         'I-ARG1', 'I-ARG1', 'B-V', 'B-ARG2', 'O', 'O']

        fields = instances[2].fields
        tokens = fields["metadata"]["words"]
        assert tokens == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month', 'after',
                          'four', 'months', 'of', 'hearings', '.']
        assert fields["verb_indicator"].labels[3] == 1
        assert fields["tags"].labels == ['O', 'B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARGM-TMP',
                                         'I-ARGM-TMP', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP',
                                         'I-ARGM-TMP', 'I-ARGM-TMP', 'O', 'O']


        fields = instances[3].fields
        tokens = fields["metadata"]["words"]
        assert tokens == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month', 'after',
                          'four', 'months', 'of', 'hearings', '.']
        assert fields["verb_indicator"].labels[12] == 1
        assert fields["tags"].labels == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                         'O', 'O', 'O', 'O', 'B-V', 'O', 'O']

        # Tests a sentence with no verbal predicates.
        fields = instances[4].fields
        tokens = fields["metadata"]["words"]
        assert tokens == ["Denise", "Dillon", "Headline", "News", "."]
        assert fields["verb_indicator"].labels == [0, 0, 0, 0, 0, 0, 0]
        assert fields["tags"].labels == ['O', 'O', 'O', 'O', 'O', 'O', 'O']
