# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestSrlReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = SrlReader(lazy=lazy)
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'conll_2012' / 'subdomain')
        instances = ensure_list(instances)

        fields = instances[0].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["Mali", "government", "officials", "say", "the", "woman", "'s",
                          "confession", "was", "forced", "."]
        assert fields["verb_indicator"].labels[3] == 1
        assert fields["tags"].labels == ['B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                                         'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] == tokens[3]
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

        fields = instances[1].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["Mali", "government", "officials", "say", "the", "woman", "'s",
                          "confession", "was", "forced", "."]
        assert fields["verb_indicator"].labels[8] == 1
        assert fields["tags"].labels == ['O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1',
                                         'I-ARG1', 'I-ARG1', 'B-V', 'B-ARG2', 'O']
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] == tokens[8]
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

        fields = instances[2].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month', 'after',
                          'four', 'months', 'of', 'hearings', '.']
        assert fields["verb_indicator"].labels[2] == 1
        assert fields["tags"].labels == ['B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARGM-TMP',
                                         'I-ARGM-TMP', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP',
                                         'I-ARGM-TMP', 'I-ARGM-TMP', 'O']
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] == tokens[2]
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

        fields = instances[3].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month', 'after',
                          'four', 'months', 'of', 'hearings', '.']
        assert fields["verb_indicator"].labels[11] == 1
        assert fields["tags"].labels == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-V', 'O']
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] == tokens[11]
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

        # Tests a sentence with no verbal predicates.
        fields = instances[4].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["Denise", "Dillon", "Headline", "News", "."]
        assert fields["verb_indicator"].labels == [0, 0, 0, 0, 0]
        assert fields["tags"].labels == ['O', 'O', 'O', 'O', 'O']
        assert fields["metadata"].metadata["words"] == tokens
        assert fields["metadata"].metadata["verb"] is None
        assert fields["metadata"].metadata["gold_tags"] == fields["tags"].labels

    def test_srl_reader_can_filter_by_domain(self):

        conll_reader = SrlReader(domain_identifier="subdomain2")
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'conll_2012')
        instances = ensure_list(instances)
        # If we'd included the folder, we'd have 9 instances.
        assert len(instances) == 2
