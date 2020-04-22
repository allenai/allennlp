import inspect
import os

import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import push_python_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.samplers import Sampler, BatchSampler
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.regularizers.regularizer import Regularizer


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
        assert "fake" not in base_class.list_available()

        @base_class.register("fake")
        class Fake(base_class):

            pass

        assert base_class.by_name("fake") == Fake

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

            @base_class.register("fake")
            class FakeAlternate(base_class):

                pass

        # Registering under a name that already exists should overwrite
        # if exist_ok=True.
        @base_class.register("fake", exist_ok=True)  # noqa
        class FakeAlternate(base_class):

            pass

        assert base_class.by_name("fake") == FakeAlternate

        del Registrable._registry[base_class]["fake"]

    # TODO(mattg): maybe move all of these into tests for the base class?

    def test_registry_has_builtin_samplers(self):
        assert Sampler.by_name("random").__name__ == "RandomSampler"
        assert Sampler.by_name("sequential").__name__ == "SequentialSampler"
        assert BatchSampler.by_name("bucket").__name__ == "BucketBatchSampler"

    def test_registry_has_builtin_tokenizers(self):
        assert Tokenizer.by_name("spacy").__name__ == "SpacyTokenizer"
        assert Tokenizer.by_name("character").__name__ == "CharacterTokenizer"

    def test_registry_has_builtin_token_indexers(self):
        assert TokenIndexer.by_name("single_id").__name__ == "SingleIdTokenIndexer"
        assert TokenIndexer.by_name("characters").__name__ == "TokenCharactersIndexer"

    def test_registry_has_builtin_regularizers(self):
        assert Regularizer.by_name("l1").__name__ == "L1Regularizer"
        assert Regularizer.by_name("l2").__name__ == "L2Regularizer"

    def test_registry_has_builtin_token_embedders(self):
        assert TokenEmbedder.by_name("embedding").__name__ == "Embedding"
        assert TokenEmbedder.by_name("character_encoding").__name__ == "TokenCharactersEncoder"

    def test_registry_has_builtin_text_field_embedders(self):
        assert TextFieldEmbedder.by_name("basic").__name__ == "BasicTextFieldEmbedder"

    def test_implicit_include_package(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / "testpackage"
        packagedir.mkdir()
        (packagedir / "__init__.py").touch()

        # And add that directory to the path
        with push_python_path(self.TEST_DIR):
            # Write out a duplicate dataset reader there, but registered under a different name.
            reader = DatasetReader.by_name("text_classification_json")

            with open(inspect.getabsfile(reader)) as f:
                code = f.read().replace(
                    """@DatasetReader.register("text_classification_json")""",
                    """@DatasetReader.register("text_classification_json-fake")""",
                )

            with open(os.path.join(packagedir, "reader.py"), "w") as f:
                f.write(code)

            # Fails to import by registered name
            with pytest.raises(ConfigurationError) as exc:
                DatasetReader.by_name("text_classification_json-fake")
                assert "is not a registered name" in str(exc.value)

            # Fails to import with wrong module name
            with pytest.raises(ConfigurationError) as exc:
                DatasetReader.by_name(
                    "testpackage.text_classification_json.TextClassificationJsonReader"
                )
                assert "unable to import module" in str(exc.value)

            # Fails to import with wrong class name
            with pytest.raises(ConfigurationError):
                DatasetReader.by_name("testpackage.reader.FakeReader")
                assert "unable to find class" in str(exc.value)

            # Imports successfully with right fully qualified name
            duplicate_reader = DatasetReader.by_name(
                "testpackage.reader.TextClassificationJsonReader"
            )
            assert duplicate_reader.__name__ == "TextClassificationJsonReader"
