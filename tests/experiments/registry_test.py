# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.experiments import Registry
from allennlp.testing.test_case import AllenNlpTestCase


class TestRegistry(AllenNlpTestCase):
    def test_registry_has_builtin_readers(self):
        assert Registry.get_dataset_reader('snli').__name__ == 'SnliReader'
        assert Registry.get_dataset_reader('sequence_tagging').__name__ == 'SequenceTaggingDatasetReader'
        assert Registry.get_dataset_reader('language_modeling').__name__ == 'LanguageModelingReader'
        assert Registry.get_dataset_reader('squad_sentence_selection').__name__ == 'SquadSentenceSelectionReader'

    def test_register_dataset_reader_fails_on_duplicate(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            @Registry.register_dataset_reader("snli")
            class NewSnliReader:
                pass

    def test_register_dataset_reader_adds_new_reader_with_decorator(self):
        assert 'fake' not in Registry.get_dataset_readers()
        @Registry.register_dataset_reader('fake')
        class Fake:
            pass
        assert Registry.get_dataset_reader('fake') == Fake
        del Registry._dataset_readers['fake']  # pylint: disable=protected-access
