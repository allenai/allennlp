# pylint: disable=invalid-name,no-self-use
import argparse
import os

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.make_vocab import MakeVocab, make_vocab_from_args, make_vocab_from_params
from allennlp.data import Vocabulary

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

    def test_make_vocab_succeeds_without_vocabulary_key(self):
        vocab_path = self.TEST_DIR / 'vocabulary'
        make_vocab_from_params(self.params, vocab_path)

    def test_make_vocab_makes_vocab(self):
        vocab_path = self.TEST_DIR / 'vocabulary'

        make_vocab_from_params(self.params, vocab_path)

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

        make_vocab_from_params(self.params, vocab_path)

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
        existing_vocab_path = self.TEST_DIR / 'vocabulary_existing'
        extended_vocab_path = self.TEST_DIR / 'vocabulary_extended'

        vocab = Vocabulary()
        vocab.add_token_to_namespace('some_weird_token_1', namespace='tokens')
        vocab.add_token_to_namespace('some_weird_token_2', namespace='tokens')
        os.makedirs(existing_vocab_path, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)

        self.params['vocabulary'] = {}
        self.params['vocabulary']['directory_path'] = existing_vocab_path
        self.params['vocabulary']['extend'] = True
        self.params['vocabulary']['min_count'] = {"tokens" : 3}
        make_vocab_from_params(self.params, extended_vocab_path)

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
        existing_vocab_path = self.TEST_DIR / 'vocabulary_existing'
        extended_vocab_path = self.TEST_DIR / 'vocabulary_extended'

        vocab = Vocabulary()
        vocab.add_token_to_namespace('some_weird_token_1', namespace='tokens')
        vocab.add_token_to_namespace('some_weird_token_2', namespace='tokens')
        os.makedirs(existing_vocab_path, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)

        self.params['vocabulary'] = {}
        self.params['vocabulary']['directory_path'] = existing_vocab_path
        self.params['vocabulary']['extend'] = False
        make_vocab_from_params(self.params, extended_vocab_path)

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
