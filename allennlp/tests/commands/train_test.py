import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from typing import Iterable

import pytest
import torch

from allennlp.commands.train import Train, TrainModel, train_model, train_model_from_args
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.models import load_archive
from allennlp.models.archival import CONFIG_NAME
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file

SEQUENCE_TAGGING_DATA_PATH = str(AllenNlpTestCase.FIXTURES_ROOT / "data" / "sequence_tagging.tsv")


class TestTrain(AllenNlpTestCase):
    def test_train_model(self):
        params = lambda: Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )

        train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, "test_train_model"))

        # It's OK if serialization dir exists but is empty:
        serialization_dir2 = os.path.join(self.TEST_DIR, "empty_directory")
        assert not os.path.exists(serialization_dir2)
        os.makedirs(serialization_dir2)
        train_model(params(), serialization_dir=serialization_dir2)

        # It's not OK if serialization dir exists and has junk in it non-empty:
        serialization_dir3 = os.path.join(self.TEST_DIR, "non_empty_directory")
        assert not os.path.exists(serialization_dir3)
        os.makedirs(serialization_dir3)
        with open(os.path.join(serialization_dir3, "README.md"), "w") as f:
            f.write("TEST")

        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=serialization_dir3)

        # It's also not OK if serialization dir is a real serialization dir:
        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, "test_train_model"))

        # But it's OK if serialization dir exists and --recover is specified:
        train_model(
            params(),
            serialization_dir=os.path.join(self.TEST_DIR, "test_train_model"),
            recover=True,
        )

        # It's ok serialization dir exists and --force is specified (it will be deleted):
        train_model(
            params(), serialization_dir=os.path.join(self.TEST_DIR, "test_train_model"), force=True
        )

        # But --force and --recover cannot both be specified
        with pytest.raises(ConfigurationError):
            train_model(
                params(),
                serialization_dir=os.path.join(self.TEST_DIR, "test_train_model"),
                force=True,
                recover=True,
            )

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need multiple GPUs.")
    def test_train_model_distributed(self):

        params = lambda: Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
                "distributed": {"cuda_devices": [0, 1]},
            }
        )

        out_dir = os.path.join(self.TEST_DIR, "test_distributed_train")
        train_model(params(), serialization_dir=out_dir)

        # Check that some logs specific to distributed
        # training are where we expect.
        serialized_files = os.listdir(out_dir)
        assert "stderr_worker0.log" in serialized_files
        assert "stdout_worker0.log" in serialized_files
        assert "stderr_worker1.log" in serialized_files
        assert "stdout_worker1.log" in serialized_files
        assert "model.tar.gz" in serialized_files

        # Check we can load the seralized model
        assert load_archive(out_dir).model

    def test_distributed_raises_error_with_no_gpus(self):
        params = Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
                "distributed": {},
            }
        )
        with pytest.raises(ConfigurationError):
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "test_train_model"))

    def test_train_saves_all_keys_in_config(self):
        params = Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "pytorch_seed": 42,
                "numpy_seed": 42,
                "random_seed": 42,
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )

        serialization_dir = os.path.join(self.TEST_DIR, "test_train_model")
        params_as_dict = (
            params.as_ordered_dict()
        )  # Do it here as train_model will pop all the values.
        train_model(params, serialization_dir=serialization_dir)

        config_path = os.path.join(serialization_dir, CONFIG_NAME)
        with open(config_path, "r") as config:
            saved_config_as_dict = OrderedDict(json.load(config))
        assert params_as_dict == saved_config_as_dict

    def test_error_is_throw_when_cuda_device_is_not_available(self):
        params = Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": "allennlp/tests/fixtures/data/sequence_tagging.tsv",
                "validation_data_path": "allennlp/tests/fixtures/data/sequence_tagging.tsv",
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                    "num_epochs": 2,
                    "cuda_device": torch.cuda.device_count(),
                    "optimizer": "adam",
                },
            }
        )

        with pytest.raises(ConfigurationError, match="Experiment specified"):
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "test_train_model"))

    def test_train_with_test_set(self):
        params = Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "test_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "evaluate_on_test": True,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "train_with_test_set"))

    def test_train_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title="Commands", metavar="")
        Train().add_subparser(subparsers)

        for serialization_arg in ["-s", "--serialization-dir"]:
            raw_args = ["train", "path/to/params", serialization_arg, "serialization_dir"]

            args = parser.parse_args(raw_args)

            assert args.func == train_model_from_args
            assert args.param_path == "path/to/params"
            assert args.serialization_dir == "serialization_dir"

        # config is required
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(["train", "-s", "serialization_dir"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

        # serialization dir is required
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(["train", "path/to/params"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

    def test_train_model_can_instantiate_from_params(self):
        params = Params.from_file(self.FIXTURES_ROOT / "simple_tagger" / "experiment.json")

        # Can instantiate from base class params
        TrainModel.from_params(
            params=params, serialization_dir=self.TEST_DIR, local_rank=0, batch_weight_key=""
        )

    def test_train_can_fine_tune_model_from_archive(self):
        params = Params.from_file(
            self.FIXTURES_ROOT / "basic_classifier" / "experiment_from_archive.jsonnet"
        )
        train_loop = TrainModel.from_params(
            params=params, serialization_dir=self.TEST_DIR, local_rank=0, batch_weight_key=""
        )
        train_loop.run()

        # This is checking that the vocabulary actually got extended.  This token was not in the
        # original training data for the model that we are fine-tuning, but is in the new training
        # data, and the vocabulary settings in the experiment config are such that this token should
        # have been added.
        assert "journalism" in train_loop.model.vocab.get_token_to_index_vocabulary()


@DatasetReader.register("lazy-test")
class LazyFakeReader(DatasetReader):
    def __init__(self) -> None:
        super().__init__(lazy=True)
        self.reader = DatasetReader.from_params(Params({"type": "sequence_tagging", "lazy": True}))

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads some data from the `file_path` and returns the instances.
        """
        return self.reader.read(file_path)


class TestTrainOnLazyDataset(AllenNlpTestCase):
    def test_train_model(self):
        params = Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "lazy-test"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "train_lazy_model"))

    def test_train_with_test_set(self):
        params = Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "lazy-test"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "test_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "evaluate_on_test": True,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "lazy_test_set"))

    def test_train_nograd_regex(self):
        params_get = lambda: Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )
        serialization_dir = os.path.join(self.TEST_DIR, "test_train_nograd")
        regex_lists = [[], [".*text_field_embedder.*"], [".*text_field_embedder.*", ".*encoder.*"]]
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
        with pytest.raises(Exception):
            train_model(params, serialization_dir=serialization_dir)


class TestFineTune(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.model_archive = str(
            self.FIXTURES_ROOT / "decomposable_attention" / "serialization" / "model.tar.gz"
        )
        self.config_file = str(self.FIXTURES_ROOT / "decomposable_attention" / "experiment.json")
        self.serialization_dir = str(self.TEST_DIR / "fine_tune")

    def test_fine_tune_nograd_regex(self):
        original_model = load_archive(self.model_archive).model
        name_parameters_original = dict(original_model.named_parameters())
        regex_lists = [
            [],
            [".*attend_feedforward.*", ".*token_embedder.*"],
            [".*compare_feedforward.*"],
        ]
        for regex_list in regex_lists:
            params = Params.from_file(self.config_file)
            params["trainer"]["no_grad"] = regex_list
            shutil.rmtree(self.serialization_dir, ignore_errors=True)
            tuned_model = train_model(
                model=original_model, params=params, serialization_dir=self.serialization_dir
            )
            # If regex is matched, parameter name should have requires_grad False
            # If regex is matched, parameter name should have same requires_grad
            # as the originally loaded model
            for name, parameter in tuned_model.named_parameters():
                if any(re.search(regex, name) for regex in regex_list):
                    assert not parameter.requires_grad
                else:
                    assert parameter.requires_grad == name_parameters_original[name].requires_grad
        # If all parameters have requires_grad=False, then error.
        with pytest.raises(Exception) as _:
            params = Params.from_file(self.config_file)
            params["trainer"]["no_grad"] = ["*"]
            shutil.rmtree(self.serialization_dir, ignore_errors=True)
            train_model(
                model=original_model, params=params, serialization_dir=self.serialization_dir
            )
