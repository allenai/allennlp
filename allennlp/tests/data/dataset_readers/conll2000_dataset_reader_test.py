# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.data.dataset_readers.conll2000 import Conll2000DatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestConll2000Reader():
    @pytest.mark.parametrize("lazy", (True, False))
    @pytest.mark.parametrize("coding_scheme", ('BIO', 'BIOUL'))
    def test_read_from_file(self, lazy, coding_scheme):
        conll_reader = Conll2000DatasetReader(lazy=lazy, coding_scheme=coding_scheme)
        instances = conll_reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'conll2000.txt'))
        instances = ensure_list(instances)
        assert len(instances) == 2

        if coding_scheme == 'BIO':
            expected_labels = [
                    'B-NP', 'B-PP', 'B-NP', 'I-NP', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'B-NP',
                    'I-NP', 'I-NP', 'B-SBAR', 'B-NP', 'I-NP', 'B-PP', 'B-NP', 'O', 'B-ADJP', 'B-PP',
                    'B-NP', 'B-NP', 'O', 'B-VP', 'I-VP', 'I-VP', 'B-NP', 'I-NP', 'I-NP', 'B-PP',
                    'B-NP', 'I-NP', 'I-NP', 'B-NP', 'I-NP', 'I-NP', 'O']
        else:
            expected_labels = [
                    'U-NP', 'U-PP', 'B-NP', 'L-NP', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'L-VP', 'B-NP', 'I-NP',
                    'L-NP', 'U-SBAR', 'B-NP', 'L-NP', 'U-PP', 'U-NP', 'O', 'U-ADJP', 'U-PP', 'U-NP', 'U-NP',
                    'O', 'B-VP', 'I-VP', 'L-VP', 'B-NP', 'I-NP', 'L-NP', 'U-PP', 'B-NP', 'I-NP', 'L-NP', 'B-NP',
                    'I-NP', 'L-NP', 'O']

        fields = instances[0].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['Confidence', 'in', 'the', 'pound', 'is', 'widely', 'expected', 'to', 'take',
                          'another', 'sharp', 'dive', 'if', 'trade', 'figures', 'for', 'September', ',',
                          'due', 'for', 'release', 'tomorrow', ',', 'fail', 'to', 'show', 'a', 'substantial',
                          'improvement', 'from', 'July', 'and', 'August', "'s", 'near-record', 'deficits', '.']
        assert fields["tags"].labels == expected_labels

        if coding_scheme == 'BIO':
            expected_labels = [
                    'O', 'B-PP', 'B-NP', 'I-NP', 'B-NP', 'I-NP', 'B-NP', 'I-NP', 'I-NP', 'B-PP', 'B-NP',
                    'I-NP', 'I-NP', 'I-NP', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'B-NP', 'I-NP', 'B-PP', 'B-NP',
                    'B-PP', 'B-NP', 'I-NP', 'I-NP', 'O']
        else:
            expected_labels = [
                    'O', 'U-PP', 'B-NP', 'L-NP', 'B-NP', 'L-NP', 'B-NP', 'I-NP', 'L-NP', 'U-PP', 'B-NP',
                    'I-NP', 'I-NP', 'L-NP', 'B-VP', 'I-VP', 'I-VP', 'L-VP', 'B-NP', 'L-NP', 'U-PP', 'U-NP',
                    'U-PP', 'B-NP', 'I-NP', 'L-NP', 'O']

        fields = instances[1].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['Chancellor', 'of', 'the', 'Exchequer', 'Nigel', 'Lawson', "'s", 'restated',
                          'commitment', 'to', 'a', 'firm', 'monetary', 'policy', 'has', 'helped', 'to',
                          'prevent', 'a', 'freefall', 'in', 'sterling', 'over', 'the', 'past', 'week', '.']
        assert fields["tags"].labels == expected_labels
