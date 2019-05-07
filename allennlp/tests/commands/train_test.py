# pylint: disable=invalid-name,no-self-use,protected-access
import argparse
from typing import Iterable
import os
import shutil
import re

import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import Train, train_model, train_model_from_args
from allennlp.data import DatasetReader, Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import load_archive

SEQUENCE_TAGGING_DATA_PATH = str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv')

class TestTrain(AllenNlpTestCase):

    def test_train_model(self):
        params = lambda: Params({
                "model": {
                        "type": "simple_tagger",
                        "text_field_embedder": {
                                "token_embedders": {
                                        "tokens": {
                                                "type": "embedding",
                                                "embedding_dim": 5
                                        }
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
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
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

        # But it's OK if serialization dir exists and --recover is specified:
        train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'), recover=True)

        # It's ok serialization dir exists and --force is specified (it will be deleted):
        train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'), force=True)

        # But --force and --recover cannot both be specified
        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'),
                        force=True, recover=True)

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

        with pytest.raises(ConfigurationError, match="Experiment specified"):
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'))

    def test_train_with_test_set(self):
        params = Params({
                "model": {
                        "type": "simple_tagger",
                        "text_field_embedder": {
                                "token_embedders": {
                                        "tokens": {
                                                "type": "embedding",
                                                "embedding_dim": 5
                                        }
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
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "test_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
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
        self.reader = DatasetReader.from_params(Params({'type': 'sequence_tagging', 'lazy': True}))

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads some data from the `file_path` and returns the instances.
        """
        return self.reader.read(file_path)


class TestTrainOnLazyDataset(AllenNlpTestCase):
    def test_train_model(self):
        params = Params({
                "model": {
                        "type": "simple_tagger",
                        "text_field_embedder": {
                                "token_embedders": {
                                        "tokens": {
                                                "type": "embedding",
                                                "embedding_dim": 5
                                        }
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
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
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
                                "token_embedders": {
                                        "tokens": {
                                                "type": "embedding",
                                                "embedding_dim": 5
                                        }
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
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "test_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "evaluate_on_test": True,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'lazy_test_set'))

    def test_train_nograd_regex(self):
        params_get = lambda: Params({
                "model": {
                        "type": "simple_tagger",
                        "text_field_embedder": {
                                "token_embedders": {
                                        "tokens": {
                                                "type": "embedding",
                                                "embedding_dim": 5
                                        }
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
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })
        serialization_dir = os.path.join(self.TEST_DIR, 'test_train_nograd')
        regex_lists = [[],
                       [".*text_field_embedder.*"],
                       [".*text_field_embedder.*", ".*encoder.*"]]
        for regex_list in regex_lists:
            params = params_get()
            params["trainer"]["no_grad"] = regex_list
            shutil.rmtree(serialization_dir, ignore_errors=True)
            model = train_model(params, serialization_dir=serialization_dir)
            # If regex is matched, parameter name should have requires_grad False
            # Or else True
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in regex_list):
                    assert not parameter.requires_grad
                else:
                    assert parameter.requires_grad
        # If all parameters have requires_grad=False, then error.
        params = params_get()
        params["trainer"]["no_grad"] = ["*"]
        shutil.rmtree(serialization_dir, ignore_errors=True)
        with pytest.raises(Exception) as _:
            model = train_model(params, serialization_dir=serialization_dir)

    def test_vocab_extended_model_with_transferred_embedder_is_loadable(self):
        # Train on snli2 but load text_field_embedder and vocab from the model
        # trained on snli (snli2 has one extra token over snli).
        # Make sure (1) embedding extension happens implicitly.
        #           (2) model dumped in such a way is loadable.
        # (1) corresponds to model.extend_embedder_vocab() in trainer.py
        # (2) corresponds to model.extend_embedder_vocab() in model.py
        config_file = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'experiment.json')
        model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        serialization_dir = str(self.TEST_DIR / 'train')

        params = Params.from_file(config_file).as_dict()

        snli_vocab_path = str(self.FIXTURES_ROOT / 'data' / 'snli_vocab')
        params["train_data_path"] = str(self.FIXTURES_ROOT / 'data' / 'snli2.jsonl')
        params["model"]["text_field_embedder"] = {
                "_pretrained": {
                        "archive_file": model_archive,
                        "module_path": "_text_field_embedder"
                }
        }
        params["vocabulary"] = {
                "directory_path": snli_vocab_path,
                "extend": True
        }

        original_vocab = Vocabulary.from_files(snli_vocab_path)

        original_model = load_archive(model_archive).model
        original_weight = original_model._text_field_embedder.token_embedder_tokens.weight

        transferred_model = train_model(Params(params), serialization_dir=serialization_dir)

        assert original_vocab.get_vocab_size("tokens") == 24
        assert transferred_model.vocab.get_vocab_size("tokens") == 25

        extended_weight = transferred_model._text_field_embedder.token_embedder_tokens.weight
        assert original_weight.shape[0] + 1 == extended_weight.shape[0] == 25
        assert torch.all(original_weight == extended_weight[:24, :])

        # Check that such a dumped model is loadable
        # self.serialization_dir = self.TEST_DIR / 'fine_tune'
        load_archive(str(self.TEST_DIR / 'train' / "model.tar.gz"))
