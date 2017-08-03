import argparse

from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.commands.train import train_model, add_subparser, train_model_from_file


class TestTrain(AllenNlpTestCase):
    def test_train_model(self):
        trainer_params = {
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
                "train_data_path": 'tests/fixtures/sequence_tagging_example.tsv',
                "iterator": {"type": "basic", "batch_size": 2},
                "optimizer": "adam",
                "num_epochs": 2
        }
        train_model(trainer_params)

    def test_train_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        add_subparser(subparsers)

        raw_args = ["train", "path/to/params"]

        args = parser.parse_args(raw_args)

        assert args.func == train_model_from_file
        assert args.param_path == "path/to/params"

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            args = parser.parse_args(["train"])
            assert cm.exception.code == 2  # argparse code for incorrect usage
