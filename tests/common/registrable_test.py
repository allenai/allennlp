# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import Type

import pytest

import torch
import torch.nn.init
from allennlp.common.checks import ConfigurationError
from allennlp.common import Registrable
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.data.dataset_reader import DatasetReader
from allennlp.data.data_iterator import DataIterator
from allennlp.data.tokenizer import Tokenizer
from allennlp.data.token_indexer import TokenIndexer
from allennlp.training.regularizer import Regularizer
from allennlp.training.initializer import Initializer
from allennlp.modules.token_embedder import TokenEmbedder
from allennlp.modules.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.similarity_function import SimilarityFunction


class TestRegistrable(AllenNlpTestCase):

    # Helper functions to make the tests less repetitive
    @staticmethod
    def registrable_helper(base_class: Type[Registrable]):
        """
        This function tests all of the things that should be hooked up correctly given a collection
        of related registry functions:

            1. The decorator should add things to the list.
            2. The decorator should crash when adding a duplicate.
            3. If a default is given, it should show up first in the list.

        What we don't test here is that built-in items are registered correctly.  You should test
        that yourself in a separate test method.
        """
        assert 'fake' not in base_class.list_available()

        @base_class.register('fake')
        class Fake(base_class):
            pass

        assert base_class.by_name('fake') == Fake

        default = base_class.default_implementation
        if default is not None:
            assert base_class.list_available()[0] == default
            base_class.default_implementation = "fake"
            assert base_class.list_available()[0] == "fake"

            with pytest.raises(ConfigurationError):
                base_class.default_implementation = "not present"
                base_class.list_available()
            base_class.default_implementation = default

        del Registrable._registry[base_class]['fake']  # pylint: disable=protected-access



    # Dataset readers

    def test_registry_has_builtin_readers(self):
        assert DatasetReader.by_name('snli').__name__ == 'SnliReader'
        assert DatasetReader.by_name('sequence_tagging').__name__ == 'SequenceTaggingDatasetReader'
        assert DatasetReader.by_name('language_modeling').__name__ == 'LanguageModelingReader'
        assert DatasetReader.by_name('squad_sentence_selection').__name__ == 'SquadSentenceSelectionReader'

    def test_dataset_readers_use_correct_fields(self):
        self.registrable_helper(DatasetReader)

    # Data iterators

    def test_registry_has_builtin_iterators(self):
        assert DataIterator.by_name('adaptive').__name__ == 'AdaptiveIterator'
        assert DataIterator.by_name('basic').__name__ == 'BasicIterator'
        assert DataIterator.by_name('bucket').__name__ == 'BucketIterator'

    def test_data_iterators_use_correct_fields(self):
        self.registrable_helper(DataIterator)

    # Tokenizers

    def test_registry_has_builtin_tokenizers(self):
        assert Tokenizer.by_name('word').__name__ == 'WordTokenizer'
        assert Tokenizer.by_name('character').__name__ == 'CharacterTokenizer'

    def test_tokenizers_use_correct_fields(self):
        self.registrable_helper(Tokenizer)

    # Token indexers

    def test_registry_has_builtin_token_indexers(self):
        assert TokenIndexer.by_name('single_id').__name__ == 'SingleIdTokenIndexer'
        assert TokenIndexer.by_name('characters').__name__ == 'TokenCharactersIndexer'

    def test_token_indexers_use_correct_fields(self):
        self.registrable_helper(TokenIndexer)

    # Regularizers

    def test_registry_has_builtin_regularizers(self):
        assert Regularizer.by_name('l1').__name__ == 'L1Regularizer'
        assert Regularizer.by_name('l2').__name__ == 'L2Regularizer'

    def test_regularizers_use_correct_fields(self):
        self.registrable_helper(Regularizer)

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
            assert Initializer.by_name(key) == value

    def test_initializers_use_correct_fields(self):
        self.registrable_helper(Initializer)

    # Token embedders

    def test_registry_has_builtin_token_embedders(self):
        assert TokenEmbedder.by_name("embedding").__name__ == 'Embedding'
        assert TokenEmbedder.by_name("character_encoding").__name__ == 'TokenCharactersEncoder'

    def test_token_embedders_use_correct_fields(self):
        self.registrable_helper(TokenEmbedder)

    # Text field embedders

    def test_registry_has_builtin_text_field_embedders(self):
        assert TextFieldEmbedder.by_name("basic").__name__ == 'BasicTextFieldEmbedder'

    def test_text_field_embedders_use_correct_fields(self):
        self.registrable_helper(TextFieldEmbedder)

    # Seq2Seq encoders

    def test_registry_has_builtin_seq2seq_encoders(self):
        # pylint: disable=protected-access
        assert Seq2SeqEncoder.by_name('gru')._module_class.__name__ == 'GRU'
        assert Seq2SeqEncoder.by_name('lstm')._module_class.__name__ == 'LSTM'
        assert Seq2SeqEncoder.by_name('rnn')._module_class.__name__ == 'RNN'

    def test_seq2seq_encoders_use_correct_fields(self):
        self.registrable_helper(Seq2SeqEncoder)

    # Seq2Vec encoders

    def test_registry_has_builtin_seq2vec_encoders(self):
        assert Seq2VecEncoder.by_name('cnn').__name__ == 'CnnEncoder'
        # pylint: disable=protected-access
        assert Seq2VecEncoder.by_name('gru')._module_class.__name__ == 'GRU'
        assert Seq2VecEncoder.by_name('lstm')._module_class.__name__ == 'LSTM'
        assert Seq2VecEncoder.by_name('rnn')._module_class.__name__ == 'RNN'

    def test_seq2vec_encoders_use_correct_fields(self):
        self.registrable_helper(Seq2VecEncoder)

    # Similarity functions

    def test_registry_has_builtin_similarity_functions(self):
        assert SimilarityFunction.by_name("dot_product").__name__ == 'DotProductSimilarity'
        assert SimilarityFunction.by_name("bilinear").__name__ == 'BilinearSimilarity'
        assert SimilarityFunction.by_name("linear").__name__ == 'LinearSimilarity'
        assert SimilarityFunction.by_name("cosine").__name__ == 'CosineSimilarity'

    def test_similarity_functions_use_correct_fields(self):
        self.registrable_helper(SimilarityFunction)
