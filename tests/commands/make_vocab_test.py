# pylint: disable=invalid-name,no-self-use
import os

import pytest

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.make_vocab import make_vocab_from_params

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
                        "stacked_encoder": {
                                "type": "lstm",
                                "input_size": 5,
                                "hidden_size": 7,
                                "num_layers": 2
                        }
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "validation_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

    def test_make_vocab_fails_without_vocabulary_key(self):
        with pytest.raises(ConfigurationError):
            make_vocab_from_params(self.params)

    def test_make_vocab_makes_vocab(self):
        vocab_path = os.path.join(self.TEST_DIR, 'vocabulary')

        self.params['vocabulary'] = {}
        self.params['vocabulary']['directory_path'] = vocab_path

        make_vocab_from_params(self.params)

        vocab_files = os.listdir(vocab_path)
        assert set(vocab_files) == {'labels.txt', 'non_padded_namespaces.txt', 'tokens.txt'}

        with open(os.path.join(vocab_path, 'tokens.txt')) as f:
            tokens = [line.strip() for line in f]

        tokens.sort()
        assert tokens == ['.', '@@UNKNOWN@@', 'animals', 'are', 'birds', 'cats', 'dogs', 'snakes']

        with open(os.path.join(vocab_path, 'labels.txt')) as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == ['N', 'V']
