# pylint: disable=invalid-name,no-self-use
import argparse
import os
import json

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import train_model, add_subparser, _train_model_from_args, _CONFIG_FILE_KEY


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
                "optimizer": "adam",
                "trainer": {
                        "num_epochs": 2,
                        "serialization_prefix": self.TEST_DIR
                }
        })

        # Write params to file for archiving purposes
        config_file = os.path.join(self.TEST_DIR, "config.json")
        with open(config_file, 'w') as outfile:
            json.dump(params.as_dict(), outfile)
        params[_CONFIG_FILE_KEY] = config_file
        train_model(params)

    def test_train_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        add_subparser(subparsers)

        raw_args = ["train", "path/to/params"]

        args = parser.parse_args(raw_args)

        assert args.func == _train_model_from_args
        assert args.param_path == "path/to/params"

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            args = parser.parse_args(["train"])
            assert cm.exception.code == 2  # argparse code for incorrect usage
