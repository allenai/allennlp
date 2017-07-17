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

    def test_registry_has_builtin_iterators(self):
        assert Registry.get_data_iterator('adaptive').__name__ == 'AdaptiveIterator'
        assert Registry.get_data_iterator('basic').__name__ == 'BasicIterator'
        assert Registry.get_data_iterator('bucket').__name__ == 'BucketIterator'

    def test_register_data_iterator_fails_on_duplicate(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            @Registry.register_data_iterator("bucket")
            class NewBucketIterator:
                pass

    def test_register_data_iterator_adds_new_iterator_with_decorator(self):
        assert 'fake' not in Registry.get_data_iterators()
        @Registry.register_data_iterator('fake')
        class Fake:
            pass
        assert Registry.get_data_iterator('fake') == Fake
        del Registry._data_iterators['fake']  # pylint: disable=protected-access

    def test_default_data_iterator_is_first_in_list(self):
        default_iterator = Registry.default_data_iterator
        assert Registry.get_data_iterators()[0] == default_iterator
        Registry.default_data_iterator = "basic"
        assert Registry.get_data_iterators()[0] == "basic"
        Registry.default_data_iterator = default_iterator
