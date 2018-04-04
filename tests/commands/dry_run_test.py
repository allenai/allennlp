# pylint: disable=invalid-name,no-self-use
import os

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.dry_run import dry_run_from_params

class TestDryRun(AllenNlpTestCase):
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
                "train_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "validation_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

    def test_dry_run_doesnt_overwrite_vocab(self):
        vocab_path = os.path.join(self.TEST_DIR, 'pre-defined-vocab')
        os.mkdir(vocab_path)
        # Put something in the vocab directory
        with open(os.path.join(vocab_path, "test.txt"), "a+") as open_file:
            open_file.write("test")

        self.params['vocabulary'] = {}
        self.params['vocabulary']['directory_path'] = vocab_path

        dry_run_from_params(self.params, self.TEST_DIR)

        # Shouldn't have been overwritten.
        predefined_vocab_files = os.listdir(vocab_path)
        assert set(predefined_vocab_files) == {'test.txt'}
        # But we should have written the created vocab to serialisation_dir/vocab:
        new_vocab_files = os.listdir(os.path.join(self.TEST_DIR, 'vocabulary'))
        assert set(new_vocab_files) == {'tokens.txt', 'non_padded_namespaces.txt', 'labels.txt'}
