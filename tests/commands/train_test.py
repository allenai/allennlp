# pylint: disable=invalid-name,no-self-use
import argparse

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import train_model, add_subparser, _train_model_from_args


class TestTrain(AllenNlpTestCase):
    def test_train_model(self):
        params = Params({
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

        train_model(params, serialization_dir=self.TEST_DIR)

    def test_train_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        add_subparser(subparsers)

        raw_args = ["train", "path/to/params", "-s", "serialization_dir"]

        args = parser.parse_args(raw_args)

        assert args.func == _train_model_from_args
        assert args.param_path == "path/to/params"
        assert args.serialization_dir == "serialization_dir"

        # config is required
        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            args = parser.parse_args(["train", "-s", "serialization_dir"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

        # serialization dir is required
        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            args = parser.parse_args(["train", "path/to/params"])
            assert cm.exception.code == 2  # argparse code for incorrect usage
