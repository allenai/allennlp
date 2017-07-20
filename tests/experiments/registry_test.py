# pylint: disable=no-self-use,invalid-name,too-many-public-methods
import pytest

import torch
import torch.nn.init
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

    # Regularizers

    def test_registry_has_builtin_regularizers(self):
        assert Registry.get_regularizer('l1').__name__ == 'L1Regularizer'
        assert Registry.get_regularizer('l2').__name__ == 'L2Regularizer'

    def test_register_regularizer_fails_on_duplicate(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            @Registry.register_regularizer("l1")
            class NewL1Regularizer:
                pass

    def test_register_regularizer_adds_new_regularizer_with_decorator(self):
        assert 'fake' not in Registry.list_regularizers()
        @Registry.register_regularizer('fake')
        class Fake:
            pass
        assert Registry.get_regularizer('fake') == Fake
        del Registry._regularizers['fake']  # pylint: disable=protected-access

    def test_default_regularizer_is_first_in_list(self):
        default_regularizer = Registry.default_regularizer
        assert Registry.list_regularizers()[0] == default_regularizer
        Registry.default_regularizer = "l1"
        assert Registry.list_regularizers()[0] == "l1"
        Registry.default_regularizer = default_regularizer

    # Initializers

    def test_registry_has_builtin_initializers(self):
        all_initializers = {
                "normal": torch.nn.init.normal,
                "uniform": torch.nn.init.uniform,
                "orthogonal": torch.nn.init.orthogonal,
                "constant": torch.nn.init.constant,
                "dirac": torch.nn.init.dirac,
                "xavier_normal": torch.nn.init.xavier_normal,
                "xavier_uniform": torch.nn.init.xavier_uniform,
                "kaiming_normal": torch.nn.init.kaiming_normal,
                "kaiming_uniform": torch.nn.init.kaiming_uniform,
                "sparse": torch.nn.init.sparse,
                "eye": torch.nn.init.eye,
        }
        for key, value in all_initializers.items():
            assert Registry.get_initializer(key) == value

    def test_register_initializers_fails_on_duplicate(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=unused-variable
            @Registry.register_initializer("normal")
            class NewL1Regularizer:
                pass

    def test_register_initializers_adds_new_initializers_with_decorator(self):
        assert 'fake' not in Registry.list_initializers()
        @Registry.register_initializer('fake')
        def fake_initializer():
            pass
        assert Registry.get_initializer('fake') == fake_initializer
        del Registry._initializers['fake']  # pylint: disable=protected-access

    def test_default_initializer_is_first_in_list(self):
        default_initializer = Registry.default_initializer
        assert Registry.list_initializers()[0] == default_initializer
        Registry.default_initializer = "orthogonal"
        assert Registry.list_initializers()[0] == "orthogonal"
        Registry.default_initializer = default_initializer

    # Token vectorizers

    def test_registry_has_builtin_token_vectorizers(self):
        assert Registry.get_token_vectorizer("embedding").__name__ == 'Embedding'

    def test_register_token_vectorizers_fails_on_duplicate(self):
        Registry.register_token_vectorizer("duplicate")(lambda: 1)
        with pytest.raises(ConfigurationError):
            Registry.register_token_vectorizer("duplicate")(lambda: 2)
        del Registry._token_vectorizers['duplicate']  # pylint: disable=protected-access

    def test_register_token_vectorizers_adds_new_token_vectorizers_with_decorator(self):
        assert 'fake' not in Registry.list_token_vectorizers()
        @Registry.register_token_vectorizer('fake')
        def fake_token_vectorizer():
            pass
        assert Registry.get_token_vectorizer('fake') == fake_token_vectorizer
        del Registry._token_vectorizers['fake']  # pylint: disable=protected-access

    def test_default_token_vectorizer_is_first_in_list(self):
        Registry.register_token_vectorizer("fake")(lambda: 1)
        default_token_vectorizer = Registry.default_token_vectorizer
        assert Registry.list_token_vectorizers()[0] == default_token_vectorizer
        Registry.default_token_vectorizer = "fake"
        assert Registry.list_token_vectorizers()[0] == "fake"
        Registry.default_token_vectorizer = default_token_vectorizer
        del Registry._token_vectorizers['fake']  # pylint: disable=protected-access

    # Token embedders

    def test_registry_has_builtin_token_embedders(self):
        assert Registry.get_token_embedder("basic").__name__ == 'BasicTokenEmbedder'

    def test_register_token_embedders_fails_on_duplicate(self):
        Registry.register_token_embedder("duplicate")(lambda: 1)
        with pytest.raises(ConfigurationError):
            Registry.register_token_embedder("duplicate")(lambda: 2)
        del Registry._token_embedders['duplicate']  # pylint: disable=protected-access

    def test_register_token_embedders_adds_new_token_embedders_with_decorator(self):
        assert 'fake' not in Registry.list_token_embedders()
        @Registry.register_token_embedder('fake')
        def fake_token_embedder():
            pass
        assert Registry.get_token_embedder('fake') == fake_token_embedder
        del Registry._token_embedders['fake']  # pylint: disable=protected-access

    def test_default_token_embedder_is_first_in_list(self):
        Registry.register_token_embedder("fake")(lambda: 1)
        default_token_embedder = Registry.default_token_embedder
        assert Registry.list_token_embedders()[0] == default_token_embedder
        Registry.default_token_embedder = "fake"
        assert Registry.list_token_embedders()[0] == "fake"
        Registry.default_token_embedder = default_token_embedder
        del Registry._token_embedders['fake']  # pylint: disable=protected-access
