# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestConll2003Reader():
    @pytest.mark.parametrize("lazy", (True, False))
    @pytest.mark.parametrize("convert_to_coding_scheme", ('BIOUL',))
    def test_read_from_file(self, lazy, convert_to_coding_scheme):
        conll_reader = Conll2003DatasetReader(lazy=lazy, convert_to_coding_scheme=convert_to_coding_scheme)
        instances = conll_reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'conll2003.txt'))
        instances = ensure_list(instances)

        if convert_to_coding_scheme == 'BIOUL':
            expected_labels = ['U-ORG', 'O', 'U-PER', 'O', 'O', 'U-LOC', 'O']
        else:
            expected_labels = ['I-ORG', 'O', 'I-PER', 'O', 'O', 'I-LOC', 'O']

        fields = instances[0].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['U.N.', 'official', 'Ekeus', 'heads', 'for', 'Baghdad', '.']
        assert fields["tags"].labels == expected_labels

        fields = instances[1].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['AI2', 'engineer', 'Joel', 'lives', 'in', 'Seattle', '.']
        assert fields["tags"].labels == expected_labels
