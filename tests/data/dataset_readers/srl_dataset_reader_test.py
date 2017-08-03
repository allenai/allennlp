# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers.semantic_role_labelling import SrlReader
from allennlp.common.testing import AllenNlpTestCase


class TestSrlReader(AllenNlpTestCase):
    def test_read_from_file(self):
        conll_reader = SrlReader()
        dataset = conll_reader.read('tests/fixtures/conll_2012/')
        instances = dataset.instances
        fields = instances[0].fields()
        assert fields["tokens"].tokens() == ["Mali", "government", "officials", "say",
                                             "the", "woman", "'s", "confession", "was", "forced", "."]
        assert fields["verb_indicator"].sequence_index() == 3
        assert fields["tags"].tags() == ['B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                                         'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']
        fields = instances[1].fields()
        assert fields["tokens"].tokens() == ["Mali", "government", "officials", "say",
                                             "the", "woman", "'s", "confession", "was", "forced", "."]
        assert fields["verb_indicator"].sequence_index() == 8
        assert fields["tags"].tags() == ['O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1',
                                         'I-ARG1', 'I-ARG1', 'B-V', 'B-ARG2', 'O']
        fields = instances[2].fields()
        assert fields["tokens"].tokens() == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month',
                                             'after', 'four', 'months', 'of', 'hearings', '.']
        assert fields["verb_indicator"].sequence_index() == 2
        assert fields["tags"].tags() == ['B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARGM-TMP',
                                         'I-ARGM-TMP', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP',
                                         'I-ARGM-TMP', 'I-ARGM-TMP', 'O']
        fields = instances[3].fields()
        assert fields["tokens"].tokens() == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month',
                                             'after', 'four', 'months', 'of', 'hearings', '.']
        assert fields["verb_indicator"].sequence_index() == 11
        assert fields["tags"].tags() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-V', 'O']

        # Tests a sentence with no verbal predicates.
        fields = instances[4].fields()
        assert fields["tokens"].tokens() == ["Denise", "Dillon", "Headline", "News", "."]
        assert fields["verb_indicator"].sequence_index() is None
        assert fields["tags"].tags() == ['O', 'O', 'O', 'O', 'O']
