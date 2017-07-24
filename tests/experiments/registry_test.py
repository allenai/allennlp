# pylint: disable=no-self-use,invalid-name,too-many-public-methods
import pytest

import torch
import torch.nn.init
from allennlp.common.checks import ConfigurationError
from allennlp.experiments import Registry
from allennlp.testing.test_case import AllenNlpTestCase


class TestRegistry(AllenNlpTestCase):

    # Helper functions to make the registry tests less repetitive

    @staticmethod
    def registry_helper(list_fn, get_fn, decorator_fn, dictionary, default_attr_name: str = None):
        """
        This function tests all of the things that should be hooked up correctly given a collection
        of related registry functions:

            1. The decorator should add things to the list.
            2. The decorator should crash when adding a duplicate.
            3. If a default is given, it should show up first in the list.

        What we don't test here is that built-in items are registered correctly.  You should test
        that yourself in a separate test method.
        """
        assert 'fake' not in list_fn()
        @decorator_fn('fake')
        def fake():
            return 1
        assert get_fn('fake') == fake
        with pytest.raises(ConfigurationError):
            decorator_fn('fake')(lambda: 2)
        if default_attr_name:
            current_default = getattr(Registry, default_attr_name)
            assert list_fn()[0] == current_default
            setattr(Registry, default_attr_name, 'fake')
            assert list_fn()[0] == 'fake'
            with pytest.raises(ConfigurationError):
                setattr(Registry, default_attr_name, 'not present')
                list_fn()
            setattr(Registry, default_attr_name, current_default)
        del dictionary['fake']

    # Dataset readers

    def test_registry_has_builtin_readers(self):
        assert Registry.get_dataset_reader('snli').__name__ == 'SnliReader'
        assert Registry.get_dataset_reader('sequence_tagging').__name__ == 'SequenceTaggingDatasetReader'
        assert Registry.get_dataset_reader('language_modeling').__name__ == 'LanguageModelingReader'
        assert Registry.get_dataset_reader('squad_sentence_selection').__name__ == 'SquadSentenceSelectionReader'

    def test_dataset_readers_use_correct_fields(self):
        self.registry_helper(Registry.list_dataset_readers,
                             Registry.get_dataset_reader,
                             Registry.register_dataset_reader,
                             Registry._dataset_readers)  # pylint: disable=protected-access

    # Data iterators

    def test_registry_has_builtin_iterators(self):
        assert Registry.get_data_iterator('adaptive').__name__ == 'AdaptiveIterator'
        assert Registry.get_data_iterator('basic').__name__ == 'BasicIterator'
        assert Registry.get_data_iterator('bucket').__name__ == 'BucketIterator'

    def test_data_iterators_use_correct_fields(self):
        self.registry_helper(Registry.list_data_iterators,
                             Registry.get_data_iterator,
                             Registry.register_data_iterator,
                             Registry._data_iterators,  # pylint: disable=protected-access
                             'default_data_iterator')

    # Tokenizers

    def test_registry_has_builtin_tokenizers(self):
        assert Registry.get_tokenizer('word').__name__ == 'WordTokenizer'
        assert Registry.get_tokenizer('character').__name__ == 'CharacterTokenizer'

    def test_tokenizers_use_correct_fields(self):
        self.registry_helper(Registry.list_tokenizers,
                             Registry.get_tokenizer,
                             Registry.register_tokenizer,
                             Registry._tokenizers,  # pylint: disable=protected-access
                             'default_tokenizer')

    # Token indexers

    def test_registry_has_builtin_token_indexers(self):
        assert Registry.get_token_indexer('single_id').__name__ == 'SingleIdTokenIndexer'
        assert Registry.get_token_indexer('characters').__name__ == 'TokenCharactersIndexer'

    def test_token_indexers_use_correct_fields(self):
        self.registry_helper(Registry.list_token_indexers,
                             Registry.get_token_indexer,
                             Registry.register_token_indexer,
                             Registry._token_indexers,  # pylint: disable=protected-access
                             'default_token_indexer')

    # Regularizers

    def test_registry_has_builtin_regularizers(self):
        assert Registry.get_regularizer('l1').__name__ == 'L1Regularizer'
        assert Registry.get_regularizer('l2').__name__ == 'L2Regularizer'

    def test_regularizers_use_correct_fields(self):
        self.registry_helper(Registry.list_regularizers,
                             Registry.get_regularizer,
                             Registry.register_regularizer,
                             Registry._regularizers,  # pylint: disable=protected-access
                             'default_regularizer')

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

    def test_initializers_use_correct_fields(self):
        self.registry_helper(Registry.list_initializers,
                             Registry.get_initializer,
                             Registry.register_initializer,
                             Registry._initializers,  # pylint: disable=protected-access
                             'default_initializer')

    # Token embedders

    def test_registry_has_builtin_token_embedders(self):
        assert Registry.get_token_embedder("embedding").__name__ == 'Embedding'

    def test_token_embedders_use_correct_fields(self):
        self.registry_helper(Registry.list_token_embedders,
                             Registry.get_token_embedder,
                             Registry.register_token_embedder,
                             Registry._token_embedders,  # pylint: disable=protected-access
                             'default_token_embedder')

    # Text field embedders

    def test_registry_has_builtin_text_field_embedders(self):
        assert Registry.get_text_field_embedder("basic").__name__ == 'BasicTextFieldEmbedder'

    def test_text_field_embedders_use_correct_fields(self):
        self.registry_helper(Registry.list_text_field_embedders,
                             Registry.get_text_field_embedder,
                             Registry.register_text_field_embedder,
                             Registry._text_field_embedders,  # pylint: disable=protected-access
                             'default_text_field_embedder')

    # Seq2Seq encoders

    def test_registry_has_builtin_seq2seq_encoders(self):
        # pylint: disable=protected-access
        assert Registry.get_seq2seq_encoder('gru')._module_class.__name__ == 'GRU'
        assert Registry.get_seq2seq_encoder('lstm')._module_class.__name__ == 'LSTM'
        assert Registry.get_seq2seq_encoder('rnn')._module_class.__name__ == 'RNN'

    def test_seq2seq_encoders_use_correct_fields(self):
        self.registry_helper(Registry.list_seq2seq_encoders,
                             Registry.get_seq2seq_encoder,
                             Registry.register_seq2seq_encoder,
                             Registry._seq2seq_encoders)  # pylint: disable=protected-access

    # Seq2Vec encoders

    def test_registry_has_builtin_seq2vec_encoders(self):
        assert Registry.get_seq2vec_encoder('cnn').__name__ == 'CnnEncoder'
        # pylint: disable=protected-access
        assert Registry.get_seq2vec_encoder('gru')._module_class.__name__ == 'GRU'
        assert Registry.get_seq2vec_encoder('lstm')._module_class.__name__ == 'LSTM'
        assert Registry.get_seq2vec_encoder('rnn')._module_class.__name__ == 'RNN'

    def test_seq2vec_encoders_use_correct_fields(self):
        self.registry_helper(Registry.list_seq2vec_encoders,
                             Registry.get_seq2vec_encoder,
                             Registry.register_seq2vec_encoder,
                             Registry._seq2vec_encoders)  # pylint: disable=protected-access
