# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.experiments import Registry
from allennlp.testing.test_case import AllenNlpTestCase


class TestRegistry(AllenNlpTestCase):

    # Dataset readers

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
        assert 'fake' not in Registry.list_dataset_readers()
        @Registry.register_dataset_reader('fake')
        class Fake:
            pass
        assert Registry.get_dataset_reader('fake') == Fake
        del Registry._dataset_readers['fake']  # pylint: disable=protected-access

    # Data iterators

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
        assert 'fake' not in Registry.list_data_iterators()
        @Registry.register_data_iterator('fake')
        class Fake:
            pass
        assert Registry.get_data_iterator('fake') == Fake
        del Registry._data_iterators['fake']  # pylint: disable=protected-access

    def test_default_data_iterator_is_first_in_list(self):
        default_iterator = Registry.default_data_iterator
        assert Registry.list_data_iterators()[0] == default_iterator
        Registry.default_data_iterator = "basic"
        assert Registry.list_data_iterators()[0] == "basic"
        with pytest.raises(ConfigurationError):
            Registry.default_data_iterator = "fake"
            Registry.list_data_iterators()
        Registry.default_data_iterator = default_iterator

    # Tokenizers

    def test_registry_has_builtin_tokenizers(self):
        assert Registry.get_tokenizer('word').__name__ == 'WordTokenizer'
        assert Registry.get_tokenizer('character').__name__ == 'CharacterTokenizer'

    def test_register_tokenizer_fails_on_duplicate(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            @Registry.register_tokenizer("word")
            class NewWordTokenizer:
                pass

    def test_register_tokenizer_adds_new_iterator_with_decorator(self):
        assert 'fake' not in Registry.list_tokenizers()
        @Registry.register_tokenizer('fake')
        class Fake:
            pass
        assert Registry.get_tokenizer('fake') == Fake
        del Registry._tokenizers['fake']  # pylint: disable=protected-access

    def test_default_tokenizer_is_first_in_list(self):
        default_iterator = Registry.default_tokenizer
        assert Registry.list_tokenizers()[0] == default_iterator
        Registry.default_tokenizer = "character"
        assert Registry.list_tokenizers()[0] == "character"
        Registry.default_tokenizer = default_iterator

    # Token indexers

    def test_registry_has_builtin_token_indexers(self):
        assert Registry.get_token_indexer('single_id').__name__ == 'SingleIdTokenIndexer'
        assert Registry.get_token_indexer('characters').__name__ == 'TokenCharactersIndexer'

    def test_register_token_indexer_fails_on_duplicate(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            @Registry.register_token_indexer("single_id")
            class NewSingleIdTokenIndexer:
                pass

    def test_register_token_indexer_adds_new_token_indexer_with_decorator(self):
        assert 'fake' not in Registry.list_token_indexers()
        @Registry.register_token_indexer('fake')
        class Fake:
            pass
        assert Registry.get_token_indexer('fake') == Fake
        del Registry._token_indexers['fake']  # pylint: disable=protected-access

    def test_default_token_indexer_is_first_in_list(self):
        default_iterator = Registry.default_token_indexer
        assert Registry.list_token_indexers()[0] == default_iterator
        Registry.default_token_indexer = "characters"
        assert Registry.list_token_indexers()[0] == "characters"
        Registry.default_token_indexer = default_iterator
