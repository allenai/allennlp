# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.common.util import ensure_list

class TestLanguageModelingDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = LanguageModelingReader(tokens_per_instance=3, lazy=lazy)

        instances = ensure_list(reader.read('tests/fixtures/data/language_modeling.txt'))
        # The last potential instance is left out, which is ok, because we don't have an end token
        # in here, anyway.
        assert len(instances) == 5

        assert [t.text for t in instances[0].fields["input_tokens"].tokens] == ["This", "is", "a"]
        assert [t.text for t in instances[0].fields["output_tokens"].tokens] == ["is", "a", "sentence"]

        assert [t.text for t in instances[1].fields["input_tokens"].tokens] == ["sentence", "for", "language"]
        assert [t.text for t in instances[1].fields["output_tokens"].tokens] == ["for", "language", "modelling"]

        assert [t.text for t in instances[2].fields["input_tokens"].tokens] == ["modelling", ".", "Here"]
        assert [t.text for t in instances[2].fields["output_tokens"].tokens] == [".", "Here", "'s"]

        assert [t.text for t in instances[3].fields["input_tokens"].tokens] == ["'s", "another", "one"]
        assert [t.text for t in instances[3].fields["output_tokens"].tokens] == ["another", "one", "for"]

        assert [t.text for t in instances[4].fields["input_tokens"].tokens] == ["for", "extra", "language"]
        assert [t.text for t in instances[4].fields["output_tokens"].tokens] == ["extra", "language", "modelling"]
