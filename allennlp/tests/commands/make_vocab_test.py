# pylint: disable=invalid-name,no-self-use
import argparse
import os

import pytest

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.make_vocab import MakeVocab, make_vocab_from_args, make_vocab_from_params
from allennlp.data import Vocabulary
from allennlp.common.checks import ConfigurationError

class TestMakeVocab(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.params = Params({
                "model": {
                        "type": "simple_tagger",
                        "text_field_embedder": {
                                "tokens": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                }
                        },
                        "encoder": {
                                "type": "lstm",
                                "input_size": 5,
                                "hidden_size": 7,
                                "num_layers": 2
                        }
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv',
                "validation_data_path": self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv',
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

    def test_make_vocab_doesnt_overwrite_vocab(self):
        vocab_path = self.TEST_DIR / 'vocabulary'
        os.mkdir(vocab_path)
        # Put something in the vocab directory
        with open(vocab_path / "test.txt", "a+") as open_file:
            open_file.write("test")
        # It should raise error if vocab dir is non-empty
        with pytest.raises(ConfigurationError):
            make_vocab_from_params(self.params, self.TEST_DIR)

    def test_make_vocab_succeeds_without_vocabulary_key(self):
        make_vocab_from_params(self.params, self.TEST_DIR)

    def test_make_vocab_makes_vocab(self):
        vocab_path = self.TEST_DIR / 'vocabulary'

        make_vocab_from_params(self.params, self.TEST_DIR)

        vocab_files = os.listdir(vocab_path)
        assert set(vocab_files) == {'labels.txt', 'non_padded_namespaces.txt', 'tokens.txt'}

        with open(vocab_path / 'tokens.txt') as f:
            tokens = [line.strip() for line in f]

        tokens.sort()
        assert tokens == ['.', '@@UNKNOWN@@', 'animals', 'are', 'birds', 'cats', 'dogs', 'snakes']

        with open(vocab_path / 'labels.txt') as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == ['N', 'V']

    def test_make_vocab_makes_vocab_with_config(self):
        vocab_path = self.TEST_DIR / 'vocabulary'

        self.params['vocabulary'] = {}
        self.params['vocabulary']['min_count'] = {"tokens" : 3}

        make_vocab_from_params(self.params, self.TEST_DIR)

        vocab_files = os.listdir(vocab_path)
        assert set(vocab_files) == {'labels.txt', 'non_padded_namespaces.txt', 'tokens.txt'}

        with open(vocab_path / 'tokens.txt') as f:
            tokens = [line.strip() for line in f]

        tokens.sort()
        assert tokens == ['.', '@@UNKNOWN@@', 'animals', 'are']

        with open(vocab_path / 'labels.txt') as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == ['N', 'V']

    def test_make_vocab_with_extension(self):
        existing_serialization_dir = self.TEST_DIR / 'existing'
        extended_serialization_dir = self.TEST_DIR / 'extended'
        existing_vocab_path = existing_serialization_dir / 'vocabulary'
        extended_vocab_path = extended_serialization_dir / 'vocabulary'

        vocab = Vocabulary()
        vocab.add_token_to_namespace('some_weird_token_1', namespace='tokens')
        vocab.add_token_to_namespace('some_weird_token_2', namespace='tokens')
        os.makedirs(existing_serialization_dir, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)

        self.params['vocabulary'] = {}
        self.params['vocabulary']['directory_path'] = existing_vocab_path
        self.params['vocabulary']['extend'] = True
        self.params['vocabulary']['min_count'] = {"tokens" : 3}
        make_vocab_from_params(self.params, extended_serialization_dir)

        vocab_files = os.listdir(extended_vocab_path)
        assert set(vocab_files) == {'labels.txt', 'non_padded_namespaces.txt', 'tokens.txt'}

        with open(extended_vocab_path / 'tokens.txt') as f:
            tokens = [line.strip() for line in f]

        assert tokens[0] == '@@UNKNOWN@@'
        assert tokens[1] == 'some_weird_token_1'
        assert tokens[2] == 'some_weird_token_2'

        tokens.sort()
        assert tokens == ['.', '@@UNKNOWN@@', 'animals', 'are',
                          'some_weird_token_1', 'some_weird_token_2']

        with open(extended_vocab_path / 'labels.txt') as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == ['N', 'V']

    def test_make_vocab_without_extension(self):
        existing_serialization_dir = self.TEST_DIR / 'existing'
        extended_serialization_dir = self.TEST_DIR / 'extended'
        existing_vocab_path = existing_serialization_dir / 'vocabulary'
        extended_vocab_path = extended_serialization_dir / 'vocabulary'

        vocab = Vocabulary()
        vocab.add_token_to_namespace('some_weird_token_1', namespace='tokens')
        vocab.add_token_to_namespace('some_weird_token_2', namespace='tokens')
        # if extend is False, its users responsibility to make sure that dataset instances
        # will be indexible by provided vocabulary. At least @@UNKNOWN@@ should be present in
        # namespace for which there could be OOV entries seen in dataset during indexing.
        # For `tokens` ns, new words will be seen but `tokens` has @@UNKNOWN@@ token.
        # but for 'labels' ns, there is no @@UNKNOWN@@ so required to add 'N', 'V' upfront.
        vocab.add_token_to_namespace('N', namespace='labels')
        vocab.add_token_to_namespace('V', namespace='labels')
        os.makedirs(existing_serialization_dir, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)

        self.params['vocabulary'] = {}
        self.params['vocabulary']['directory_path'] = existing_vocab_path
        self.params['vocabulary']['extend'] = False
        make_vocab_from_params(self.params, extended_serialization_dir)

        with open(extended_vocab_path / 'tokens.txt') as f:
            tokens = [line.strip() for line in f]

        assert tokens[0] == '@@UNKNOWN@@'
        assert tokens[1] == 'some_weird_token_1'
        assert tokens[2] == 'some_weird_token_2'
        assert len(tokens) == 3

    def test_make_vocab_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        MakeVocab().add_subparser('make-vocab', subparsers)
        for serialization_arg in ["-s", "--serialization-dir"]:
            raw_args = ["make-vocab", "path/to/params", serialization_arg, "serialization_dir"]
            args = parser.parse_args(raw_args)
            assert args.func == make_vocab_from_args
            assert args.param_path == "path/to/params"
            assert args.serialization_dir == "serialization_dir"
