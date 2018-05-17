# pylint: disable=invalid-name,no-self-use
import argparse
from typing import Iterable
import os

import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import Train, train_model, train_model_from_args
from allennlp.data import DatasetReader, Instance

class TestTrain(AllenNlpTestCase):
    def test_train_model(self):
        params = lambda: Params({
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

        train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'))

        # It's OK if serialization dir exists but is empty:
        serialization_dir2 = os.path.join(self.TEST_DIR, 'empty_directory')
        assert not os.path.exists(serialization_dir2)
        os.makedirs(serialization_dir2)
        train_model(params(), serialization_dir=serialization_dir2)

        # It's not OK if serialization dir exists and has junk in it non-empty:
        serialization_dir3 = os.path.join(self.TEST_DIR, 'non_empty_directory')
        assert not os.path.exists(serialization_dir3)
        os.makedirs(serialization_dir3)
        with open(os.path.join(serialization_dir3, 'README.md'), 'w') as f:
            f.write("TEST")

        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=serialization_dir3)

        # It's also not OK if serialization dir is a real serialization dir:
        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'))

    def test_error_is_throw_when_cuda_device_is_not_available(self):
        params = Params({
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
                        "cuda_device": torch.cuda.device_count(),
                        "optimizer": "adam"
                }
        })

        with pytest.raises(ConfigurationError,
                           message="Experiment specified a GPU but none is available;"
                                   " if you want to run on CPU use the override"
                                   " 'trainer.cuda_device=-1' in the json config file."):
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'))

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
                        "encoder": {
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

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'train_with_test_set'))

    def test_train_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        Train().add_subparser('train', subparsers)

        for serialization_arg in ["-s", "--serialization-dir"]:
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

@DatasetReader.register('lazy-test')
class LazyFakeReader(DatasetReader):
    # pylint: disable=abstract-method
    def __init__(self) -> None:
        super().__init__(lazy=True)
        self.reader = DatasetReader.from_params(Params({'type': 'sequence_tagging'}))

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads some data from the `file_path` and returns the instances.
        """
        return self.reader.read(file_path)

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
                        "encoder": {
                                "type": "lstm",
                                "input_size": 5,
                                "hidden_size": 7,
                                "num_layers": 2
                        }
                },
                "dataset_reader": {"type": "lazy-test"},
                "train_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "validation_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'train_lazy_model'))

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
                        "encoder": {
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
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'lazy_test_set'))
