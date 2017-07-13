# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers import dataset_readers, register_dataset_reader
from allennlp.data.dataset_readers.language_modeling import LanguageModelingReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.dataset_readers.squad import SquadSentenceSelectionReader
from allennlp.testing.test_case import AllenNlpTestCase


class TestDatasetReader(AllenNlpTestCase):
    # Here we're just testing the registry, making sure that it's hooking things up correctly.
    def test_registry_has_builtin_readers(self):
        assert dataset_readers['snli'] == SnliReader
        assert dataset_readers['sequence tagging'] == SequenceTaggingDatasetReader
        assert dataset_readers['language modeling'] == LanguageModelingReader
        assert dataset_readers['squad sentence selection'] == SquadSentenceSelectionReader

    def test_register_dataset_reader_fails_on_duplicate(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            @register_dataset_reader("snli")
            class NewSnliReader:
                pass

    def test_register_dataset_reader_adds_new_reader_with_decorator(self):
        assert 'fake' not in dataset_readers
        @register_dataset_reader('fake')
        class Fake:
            pass
        assert dataset_readers['fake'] == Fake
        del dataset_readers['fake']
