# pylint: disable=invalid-name,no-self-use
import argparse
from typing import Iterator
import os
import pathlib
import shutil
import sys

import pytest

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands import main
from allennlp.commands.train import Train, train_model, train_model_from_args
from allennlp.data import DatasetReader, Instance, InstanceCollection
from allennlp.data.dataset import LazyDataset

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

    def test_train_with_test_set(self):
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
                "test_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "validation_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "evaluate_on_test": True,
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
        Train().add_subparser('train', subparsers)

        for serialization_arg in ["-s", "--serialization_dir", "--serialization-dir"]:
            raw_args = ["train", "path/to/params", serialization_arg, "serialization_dir"]

            args = parser.parse_args(raw_args)

            assert args.func == train_model_from_args
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

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = os.path.join(self.TEST_DIR, 'testpackage')
        pathlib.Path(packagedir).mkdir()
        pathlib.Path(os.path.join(packagedir, '__init__.py')).touch()

        # And add that directory to the path
        sys.path.insert(0, self.TEST_DIR)

        # Write out a duplicate model there, but registered under a different name.
        from allennlp.models import simple_tagger
        with open(simple_tagger.__file__) as f:
            code = f.read().replace("""@Model.register("simple_tagger")""",
                                    """@Model.register("duplicate-test-tagger")""")

        with open(os.path.join(packagedir, 'model.py'), 'w') as f:
            f.write(code)

        # Copy fixture there too.
        shutil.copy(os.path.join(os.getcwd(), 'tests/fixtures/data/sequence_tagging.tsv'), self.TEST_DIR)
        data_path = os.path.join(self.TEST_DIR, 'sequence_tagging.tsv')

        # Write out config file
        config_path = os.path.join(self.TEST_DIR, 'config.json')
        with open(config_path, 'w') as f:
            f.write("""
                "model": {
                        "type": "duplicate-test-tagger",
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
                "train_data_path": $$$,
                "validation_data_path": $$$,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
            """.replace('$$$', data_path))

        serialization_dir = os.path.join(self.TEST_DIR, 'serialization')

        # Run train with using the non-allennlp module.
        sys.argv = ["python -m allennlp.run",
                    "train", config_path,
                    "-s", serialization_dir]

        # Shouldn't be able to find the model.
        with pytest.raises(ConfigurationError):
            main()

        # Now add the --include-package flag and it should work.
        sys.argv.extend(["--include-package", 'testpackage'])

        main()

        sys.path.remove(self.TEST_DIR)

@DatasetReader.register('lazy-test')
class LazyFakeReader(DatasetReader):
    # pylint: disable=abstract-method
    def __init__(self) -> None:
        self.reader = DatasetReader.from_params(Params({'type': 'sequence_tagging'}))

    def _generator_for_file(self, file_path: str):
        def generator() -> Iterator[Instance]:
            return (instance for instance in self.reader.read(file_path))

        return generator

    def read(self, file_path: str) -> InstanceCollection:
        """
        Actually reads some data from the `file_path` and returns a :class:`Dataset`.
        """
        return LazyDataset(self._generator_for_file(file_path))

    @classmethod
    def from_params(cls, params: Params) -> 'LazyTestReader':
        return LazyFakeReader()


class TestTrainOnLazyDataset(AllenNlpTestCase):
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
                "dataset_reader": {"type": "lazy-test"},
                "train_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "validation_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "iterator": {"type": "lazy-basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

        train_model(params, serialization_dir=self.TEST_DIR)

    def test_train_with_test_set(self):
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
                "dataset_reader": {"type": "lazy-test"},
                "train_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "test_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "validation_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "evaluate_on_test": True,
                "iterator": {"type": "lazy-basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

        train_model(params, serialization_dir=self.TEST_DIR)
