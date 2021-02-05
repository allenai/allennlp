import pytest

from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestConll2003Reader:
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("coding_scheme", ("IOB1", "BIOUL"))
    def test_read_from_file_with_deprecated_parameter(self, coding_scheme):
        conll_reader = Conll2003DatasetReader(coding_scheme=coding_scheme)
        instances = ensure_list(
            conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "conll2003.txt")
        )

        if coding_scheme == "IOB1":
            expected_labels = ["I-ORG", "O", "I-PER", "O", "O", "I-LOC", "O"]
        else:
            expected_labels = ["U-ORG", "O", "U-PER", "O", "O", "U-LOC", "O"]

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["U.N.", "official", "Ekeus", "heads", "for", "Baghdad", "."]
        assert fields["tags"].labels == expected_labels

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["AI2", "engineer", "Joel", "lives", "in", "Seattle", "."]
        assert fields["tags"].labels == expected_labels

    @pytest.mark.parametrize("convert_to_coding_scheme", (None, "BIOUL"))
    def test_read_from_file(self, convert_to_coding_scheme):
        conll_reader = Conll2003DatasetReader(convert_to_coding_scheme=convert_to_coding_scheme)
        instances = ensure_list(
            conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "conll2003.txt")
        )

        if convert_to_coding_scheme is None:
            expected_labels = ["I-ORG", "O", "I-PER", "O", "O", "I-LOC", "O"]
        else:
            expected_labels = ["U-ORG", "O", "U-PER", "O", "O", "U-LOC", "O"]

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["U.N.", "official", "Ekeus", "heads", "for", "Baghdad", "."]
        assert fields["tags"].labels == expected_labels

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["AI2", "engineer", "Joel", "lives", "in", "Seattle", "."]
        assert fields["tags"].labels == expected_labels

    def test_read_data_from_with_unsupported_coding_scheme(self):
        with pytest.raises(ConfigurationError):
            # `IOB1` is not supported in `convert_to_coding_scheme`.
            Conll2003DatasetReader(convert_to_coding_scheme="IOB1")
