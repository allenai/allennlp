# pylint: disable=no-self-use,invalid-name,too-many-public-methods
import inspect
import os
import sys

import pytest
import torch
import torch.nn.init
import torch.optim.lr_scheduler

from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn import Initializer
from allennlp.nn.regularizers.regularizer import Regularizer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler


class TestRegistrable(AllenNlpTestCase):

    def test_registrable_functionality_works(self):
        # This function tests the basic `Registrable` functionality:
        #
        #   1. The decorator should add things to the list.
        #   2. The decorator should crash when adding a duplicate (unless exist_ok=True).
        #   3. If a default is given, it should show up first in the list.
        #
        # What we don't test here is that built-in items are registered correctly.  Those are
        # tested in the other tests below.
        #
        # We'll test this with the Tokenizer class, just to have a concrete class to use, and one
        # that has a default.
        base_class = Tokenizer
        assert 'fake' not in base_class.list_available()

        @base_class.register('fake')
        class Fake(base_class):
            # pylint: disable=abstract-method
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

        # Verify that registering under a name that already exists
        # causes a ConfigurationError.
        with pytest.raises(ConfigurationError):
            @base_class.register('fake')
            class FakeAlternate(base_class):
                # pylint: disable=abstract-method
                pass

        # Registering under a name that already exists should overwrite
        # if exist_ok=True.
        @base_class.register('fake', exist_ok=True)  # pylint: disable=function-redefined
        class FakeAlternate(base_class):
            # pylint: disable=abstract-method
            pass
        assert base_class.by_name('fake') == FakeAlternate

        del Registrable._registry[base_class]['fake']  # pylint: disable=protected-access

    # TODO(mattg): maybe move all of these into tests for the base class?

    def test_registry_has_builtin_dataset_readers(self):
        assert DatasetReader.by_name('snli').__name__ == 'SnliReader'
        assert DatasetReader.by_name('sequence_tagging').__name__ == 'SequenceTaggingDatasetReader'
        assert DatasetReader.by_name('language_modeling').__name__ == 'LanguageModelingReader'
        assert DatasetReader.by_name('squad').__name__ == 'SquadReader'

    def test_registry_has_builtin_iterators(self):
        assert DataIterator.by_name('basic').__name__ == 'BasicIterator'
        assert DataIterator.by_name('bucket').__name__ == 'BucketIterator'

    def test_registry_has_builtin_tokenizers(self):
        assert Tokenizer.by_name('word').__name__ == 'WordTokenizer'
        assert Tokenizer.by_name('character').__name__ == 'CharacterTokenizer'

    def test_registry_has_builtin_token_indexers(self):
        assert TokenIndexer.by_name('single_id').__name__ == 'SingleIdTokenIndexer'
        assert TokenIndexer.by_name('characters').__name__ == 'TokenCharactersIndexer'

    def test_registry_has_builtin_regularizers(self):
        assert Regularizer.by_name('l1').__name__ == 'L1Regularizer'
        assert Regularizer.by_name('l2').__name__ == 'L2Regularizer'

    def test_registry_has_builtin_initializers(self):
        all_initializers = {
                "normal": torch.nn.init.normal_,
                "uniform": torch.nn.init.uniform_,
                "orthogonal": torch.nn.init.orthogonal_,
                "constant": torch.nn.init.constant_,
                "dirac": torch.nn.init.dirac_,
                "xavier_normal": torch.nn.init.xavier_normal_,
                "xavier_uniform": torch.nn.init.xavier_uniform_,
                "kaiming_normal": torch.nn.init.kaiming_normal_,
                "kaiming_uniform": torch.nn.init.kaiming_uniform_,
                "sparse": torch.nn.init.sparse_,
                "eye": torch.nn.init.eye_,
        }
        for key, value in all_initializers.items():
            # pylint: disable=protected-access
            assert Initializer.by_name(key)()._init_function == value

    def test_registry_has_builtin_learning_rate_schedulers(self):
        all_schedulers = {
                "step": torch.optim.lr_scheduler.StepLR,
                "multi_step": torch.optim.lr_scheduler.MultiStepLR,
                "exponential": torch.optim.lr_scheduler.ExponentialLR,
                "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau
        }
        for key, value in all_schedulers.items():
            assert LearningRateScheduler.by_name(key) == value

    def test_registry_has_builtin_token_embedders(self):
        assert TokenEmbedder.by_name("embedding").__name__ == 'Embedding'
        assert TokenEmbedder.by_name("character_encoding").__name__ == 'TokenCharactersEncoder'

    def test_registry_has_builtin_text_field_embedders(self):
        assert TextFieldEmbedder.by_name("basic").__name__ == 'BasicTextFieldEmbedder'

    def test_registry_has_builtin_seq2seq_encoders(self):
        # pylint: disable=protected-access
        assert Seq2SeqEncoder.by_name('gru')._module_class.__name__ == 'GRU'
        assert Seq2SeqEncoder.by_name('lstm')._module_class.__name__ == 'LSTM'
        assert Seq2SeqEncoder.by_name('rnn')._module_class.__name__ == 'RNN'

    def test_registry_has_builtin_seq2vec_encoders(self):
        assert Seq2VecEncoder.by_name('cnn').__name__ == 'CnnEncoder'
        # pylint: disable=protected-access
        assert Seq2VecEncoder.by_name('gru')._module_class.__name__ == 'GRU'
        assert Seq2VecEncoder.by_name('lstm')._module_class.__name__ == 'LSTM'
        assert Seq2VecEncoder.by_name('rnn')._module_class.__name__ == 'RNN'

    def test_registry_has_builtin_similarity_functions(self):
        assert SimilarityFunction.by_name("dot_product").__name__ == 'DotProductSimilarity'
        assert SimilarityFunction.by_name("bilinear").__name__ == 'BilinearSimilarity'
        assert SimilarityFunction.by_name("linear").__name__ == 'LinearSimilarity'
        assert SimilarityFunction.by_name("cosine").__name__ == 'CosineSimilarity'

    def test_implicit_include_package(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / 'testpackage'
        packagedir.mkdir()  # pylint: disable=no-member
        (packagedir / '__init__.py').touch()  # pylint: disable=no-member

        # And add that directory to the path
        sys.path.insert(0, str(self.TEST_DIR))

        # Write out a duplicate dataset reader there, but registered under a different name.
        snli_reader = DatasetReader.by_name('snli')

        with open(inspect.getabsfile(snli_reader)) as f:
            code = f.read().replace("""@DatasetReader.register("snli")""",
                                    """@DatasetReader.register("snli-fake")""")

        with open(os.path.join(packagedir, 'reader.py'), 'w') as f:
            f.write(code)

        # Fails to import by registered name
        with pytest.raises(ConfigurationError) as exc:
            DatasetReader.by_name('snli-fake')
            assert "is not a registered name" in str(exc.value)

        # Fails to import with wrong module name
        with pytest.raises(ConfigurationError) as exc:
            DatasetReader.by_name('testpackage.snli_reader.SnliFakeReader')
            assert "unable to import module" in str(exc.value)

        # Fails to import with wrong class name
        with pytest.raises(ConfigurationError):
            DatasetReader.by_name('testpackage.reader.SnliFakeReader')
            assert "unable to find class" in str(exc.value)

        # Imports successfully with right fully qualified name
        duplicate_reader = DatasetReader.by_name('testpackage.reader.SnliReader')
        assert duplicate_reader.__name__ == 'SnliReader'
