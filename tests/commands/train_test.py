import argparse
import copy
import json

import logging
import math
import os
import re
import shutil
from collections import OrderedDict, Counter
from typing import Iterable, Optional, List, Dict, Any

import pytest
import torch

from allennlp.commands.train import Train, train_model, train_model_from_args, TrainModel
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, cpu_or_gpu
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.dataloader import TensorDict
from allennlp.models import load_archive, Model
from allennlp.models.archival import CONFIG_NAME
from allennlp.training import BatchCallback, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import (
    ExponentialLearningRateScheduler,
    LearningRateScheduler,
)

SEQUENCE_TAGGING_DATA_PATH = str(AllenNlpTestCase.FIXTURES_ROOT / "data" / "sequence_tagging.tsv")
SEQUENCE_TAGGING_SHARDS_PATH = str(AllenNlpTestCase.FIXTURES_ROOT / "data" / "shards" / "*")


@BatchCallback.register("training_data_logger")
class TrainingDataLoggerBatchCallback(BatchCallback):
    def __call__(  # type: ignore
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
        if is_training:
            logger = logging.getLogger(__name__)
            for batch in batch_inputs:
                for metadata in batch["metadata"]:
                    logger.info(f"First word from training data: '{metadata['words'][0]}'")  # type: ignore


_seen_training_devices = set()


@BatchCallback.register("training_device_logger")
class TrainingDeviceLoggerBatchCallback(BatchCallback):
    def __call__(  # type: ignore
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
        global _seen_training_devices
        for tensor in trainer.model.parameters():
            _seen_training_devices.add(tensor.device)


class TestTrain(AllenNlpTestCase):
    DEFAULT_PARAMS = Params(
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
            "data_loader": {"batch_size": 2},
            "trainer": {"num_epochs": 2, "optimizer": "adam"},
        }
    )

    def test_train_model(self):
        params = lambda: copy.deepcopy(self.DEFAULT_PARAMS)

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

    @cpu_or_gpu
    def test_detect_gpu(self):
        import copy

        params = copy.deepcopy(self.DEFAULT_PARAMS)
        params["trainer"]["batch_callbacks"] = ["training_device_logger"]

        global _seen_training_devices
        _seen_training_devices.clear()
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "test_detect_gpu"))
        assert len(_seen_training_devices) == 1
        seen_training_device = next(iter(_seen_training_devices))
        if torch.cuda.device_count() == 0:
            assert seen_training_device.type == "cpu"
        else:
            assert seen_training_device.type == "cuda"

    @cpu_or_gpu
    def test_force_gpu(self):
        import copy

        params = copy.deepcopy(self.DEFAULT_PARAMS)
        params["trainer"]["batch_callbacks"] = ["training_device_logger"]
        params["trainer"]["cuda_device"] = 0

        global _seen_training_devices
        _seen_training_devices.clear()
        if torch.cuda.device_count() == 0:
            with pytest.raises(ConfigurationError):
                train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "test_force_gpu"))
        else:
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "test_force_gpu"))
            assert len(_seen_training_devices) == 1
            seen_training_device = next(iter(_seen_training_devices))
            assert seen_training_device.type == "cuda"

    @cpu_or_gpu
    def test_force_cpu(self):
        import copy

        params = copy.deepcopy(self.DEFAULT_PARAMS)
        params["trainer"]["batch_callbacks"] = ["training_device_logger"]
        params["trainer"]["cuda_device"] = -1

        global _seen_training_devices
        _seen_training_devices.clear()
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "test_force_cpu"))
        assert len(_seen_training_devices) == 1
        seen_training_device = next(iter(_seen_training_devices))
        assert seen_training_device.type == "cpu"

    @cpu_or_gpu
    def test_train_model_distributed(self):
        if torch.cuda.device_count() >= 2:
            devices = [0, 1]
        else:
            devices = [-1, -1]

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
                "data_loader": {"batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
                "distributed": {"cuda_devices": devices},
            }
        )

        out_dir = os.path.join(self.TEST_DIR, "test_distributed_train")
        train_model(params(), serialization_dir=out_dir)

        # Check that some logs specific to distributed
        # training are where we expect.
        serialized_files = os.listdir(out_dir)
        assert "out_worker0.log" in serialized_files
        assert "out_worker1.log" in serialized_files
        assert "model.tar.gz" in serialized_files

        # Check we can load the serialized model
        assert load_archive(out_dir).model

    @cpu_or_gpu
    @pytest.mark.parametrize("lazy", [True, False])
    def test_train_model_distributed_with_sharded_reader(self, lazy):
        if torch.cuda.device_count() >= 2:
            devices = [0, 1]
        else:
            devices = [-1, -1]

        params = lambda: Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {
                    "type": "sharded",
                    "base_reader": {"type": "sequence_tagging"},
                    "lazy": lazy,
                },
                "train_data_path": SEQUENCE_TAGGING_SHARDS_PATH,
                "validation_data_path": SEQUENCE_TAGGING_SHARDS_PATH,
                "data_loader": {"batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
                "distributed": {"cuda_devices": devices},
            }
        )

        out_dir = os.path.join(self.TEST_DIR, "test_distributed_train")
        train_model(params(), serialization_dir=out_dir)

        # Check that some logs specific to distributed
        # training are where we expect.
        serialized_files = os.listdir(out_dir)
        assert "out_worker0.log" in serialized_files
        assert "out_worker1.log" in serialized_files
        assert "model.tar.gz" in serialized_files

        # Check we can load the serialized model
        archive = load_archive(out_dir)
        assert archive.model

        # Check that we created a vocab from all the shards.
        tokens = archive.model.vocab._token_to_index["tokens"].keys()
        assert tokens == {
            "@@PADDING@@",
            "@@UNKNOWN@@",
            "are",
            ".",
            "animals",
            "plants",
            "vehicles",
            "cats",
            "dogs",
            "snakes",
            "birds",
            "ferns",
            "trees",
            "flowers",
            "vegetables",
            "cars",
            "buses",
            "planes",
            "rockets",
        }

        # TODO: This is somewhat brittle. Make these constants in trainer.py.
        train_early = "finishing training early!"
        validation_early = "finishing validation early!"
        train_complete = "completed its entire epoch (training)."
        validation_complete = "completed its entire epoch (validation)."

        # There are three shards, but only two workers, so the first worker will have to discard some data.
        with open(os.path.join(out_dir, "out_worker0.log")) as f:
            worker0_log = f.read()
            assert train_early in worker0_log
            assert validation_early in worker0_log
            assert train_complete not in worker0_log
            assert validation_complete not in worker0_log

        with open(os.path.join(out_dir, "out_worker1.log")) as f:
            worker1_log = f.read()
            assert train_early not in worker1_log
            assert validation_early not in worker1_log
            assert train_complete in worker1_log
            assert validation_complete in worker1_log

    @cpu_or_gpu
    @pytest.mark.parametrize("lazy", [True, False])
    def test_train_model_distributed_without_sharded_reader(self, lazy: bool):
        if torch.cuda.device_count() >= 2:
            devices = [0, 1]
        else:
            devices = [-1, -1]

        num_epochs = 2
        params = lambda: Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "sequence_tagging", "lazy": lazy},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "data_loader": {"batch_size": 1},
                "trainer": {
                    "num_epochs": num_epochs,
                    "optimizer": "adam",
                    "batch_callbacks": [
                        "tests.commands.train_test.TrainingDataLoggerBatchCallback"
                    ],
                },
                "distributed": {"cuda_devices": devices},
            }
        )

        out_dir = os.path.join(self.TEST_DIR, "test_distributed_train")
        train_model(params(), serialization_dir=out_dir)

        # Check that some logs specific to distributed
        # training are where we expect.
        serialized_files = os.listdir(out_dir)
        assert "out_worker0.log" in serialized_files
        assert "out_worker1.log" in serialized_files
        assert "model.tar.gz" in serialized_files

        # Check we can load the serialized model
        archive = load_archive(out_dir)
        assert archive.model

        # Check that we created a vocab from all the shards.
        tokens = set(archive.model.vocab._token_to_index["tokens"].keys())
        assert tokens == {
            "@@PADDING@@",
            "@@UNKNOWN@@",
            "are",
            ".",
            "animals",
            "cats",
            "dogs",
            "snakes",
            "birds",
        }

        train_complete = "completed its entire epoch (training)."
        validation_complete = "completed its entire epoch (validation)."

        import re

        pattern = re.compile(r"First word from training data: '([^']*)'")
        first_word_counts = Counter()  # type: ignore
        with open(os.path.join(out_dir, "out_worker0.log")) as f:
            worker0_log = f.read()
            assert train_complete in worker0_log
            assert validation_complete in worker0_log
            for first_word in pattern.findall(worker0_log):
                first_word_counts[first_word] += 1

        with open(os.path.join(out_dir, "out_worker1.log")) as f:
            worker1_log = f.read()
            assert train_complete in worker1_log
            assert validation_complete in worker1_log
            for first_word in pattern.findall(worker1_log):
                first_word_counts[first_word] += 1

        assert first_word_counts == {
            "cats": num_epochs,
            "dogs": num_epochs,
            "snakes": num_epochs,
            "birds": num_epochs,
        }

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
                "data_loader": {"batch_size": 2},
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
                "data_loader": {"batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )

        serialization_dir = os.path.join(self.TEST_DIR, "test_train_model")
        params_as_dict = (
            params.as_ordered_dict()
        )  # Do it here as train_model will pop all the values.
        train_model(params, serialization_dir=serialization_dir)

        config_path = os.path.join(serialization_dir, CONFIG_NAME)
        with open(config_path) as config:
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
                "train_data_path": "test_fixtures/data/sequence_tagging.tsv",
                "validation_data_path": "test_fixtures/data/sequence_tagging.tsv",
                "data_loader": {"batch_size": 2},
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
                "data_loader": {"batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "train_with_test_set"))

    def test_train_number_of_steps(self):
        number_of_epochs = 2

        last_num_steps_per_epoch: Optional[int] = None

        @LearningRateScheduler.register("mock")
        class MockLRScheduler(ExponentialLearningRateScheduler):
            def __init__(self, optimizer: torch.optim.Optimizer, num_steps_per_epoch: int):
                super().__init__(optimizer)
                nonlocal last_num_steps_per_epoch
                last_num_steps_per_epoch = num_steps_per_epoch

        batch_callback_counter = 0

        @BatchCallback.register("counter")
        class CounterBatchCallback(BatchCallback):
            def __call__(
                self,
                trainer: GradientDescentTrainer,
                batch_inputs: List[List[TensorDict]],
                batch_outputs: List[Dict[str, Any]],
                epoch: int,
                batch_number: int,
                is_training: bool,
                is_master: bool,
            ) -> None:
                nonlocal batch_callback_counter
                if is_training:
                    batch_callback_counter += 1

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
                "data_loader": {"batch_size": 2},
                "trainer": {
                    "num_epochs": number_of_epochs,
                    "optimizer": "adam",
                    "learning_rate_scheduler": {"type": "mock"},
                    "batch_callbacks": ["counter"],
                },
            }
        )
        train_model(
            params.duplicate(), serialization_dir=os.path.join(self.TEST_DIR, "train_normal")
        )
        assert batch_callback_counter == last_num_steps_per_epoch * number_of_epochs
        batch_callback_counter = 0
        normal_steps_per_epoch = last_num_steps_per_epoch

        original_batch_size = params["data_loader"]["batch_size"]
        params["data_loader"]["batch_size"] = 1
        train_model(
            params.duplicate(), serialization_dir=os.path.join(self.TEST_DIR, "train_with_bs1")
        )
        assert batch_callback_counter == last_num_steps_per_epoch * number_of_epochs
        batch_callback_counter = 0
        assert normal_steps_per_epoch == math.ceil(last_num_steps_per_epoch / original_batch_size)

        params["data_loader"]["batch_size"] = original_batch_size
        params["trainer"]["num_gradient_accumulation_steps"] = 3
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, "train_with_ga"))
        assert batch_callback_counter == last_num_steps_per_epoch * number_of_epochs
        batch_callback_counter = 0
        assert math.ceil(normal_steps_per_epoch / 3) == last_num_steps_per_epoch

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
        with pytest.raises(SystemExit) as cm:
            args = parser.parse_args(["train", "-s", "serialization_dir"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

        # serialization dir is required
        with pytest.raises(SystemExit) as cm:
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

        model = Model.from_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )

        # This is checking that the vocabulary actually got extended.  The data that we're using for
        # training is different from the data we used to produce the model archive, and we set
        # parameters such that the vocab should have been extended.
        assert train_loop.model.vocab.get_vocab_size() > model.vocab.get_vocab_size()


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
                "data_loader": {"batch_size": 2},
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
                "data_loader": {"batch_size": 2},
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
                "data_loader": {"batch_size": 2},
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


class TestDryRun(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params = Params(
            {
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                    },
                    "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": str(self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"),
                "validation_data_path": str(self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"),
                "data_loader": {"batch_size": 2},
                "trainer": {"num_epochs": 2, "optimizer": "adam"},
            }
        )

    def test_dry_run_doesnt_overwrite_vocab(self):
        vocab_path = self.TEST_DIR / "vocabulary"
        os.mkdir(vocab_path)
        # Put something in the vocab directory
        with open(vocab_path / "test.txt", "a+") as open_file:
            open_file.write("test")
        # It should raise error if vocab dir is non-empty
        with pytest.raises(ConfigurationError):
            train_model(self.params, self.TEST_DIR, dry_run=True)

    def test_dry_run_without_vocabulary_key(self):
        train_model(self.params, self.TEST_DIR, dry_run=True)

    def test_dry_run_makes_vocab(self):
        vocab_path = self.TEST_DIR / "vocabulary"

        train_model(self.params, self.TEST_DIR, dry_run=True)

        vocab_files = os.listdir(vocab_path)
        assert set(vocab_files) == {
            ".lock",
            "labels.txt",
            "non_padded_namespaces.txt",
            "tokens.txt",
        }

        with open(vocab_path / "tokens.txt") as f:
            tokens = [line.strip() for line in f]

        tokens.sort()
        assert tokens == [".", "@@UNKNOWN@@", "animals", "are", "birds", "cats", "dogs", "snakes"]

        with open(vocab_path / "labels.txt") as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == ["N", "V"]

    def test_dry_run_with_extension(self):
        existing_serialization_dir = self.TEST_DIR / "existing"
        extended_serialization_dir = self.TEST_DIR / "extended"
        existing_vocab_path = existing_serialization_dir / "vocabulary"
        extended_vocab_path = extended_serialization_dir / "vocabulary"

        vocab = Vocabulary()
        vocab.add_token_to_namespace("some_weird_token_1", namespace="tokens")
        vocab.add_token_to_namespace("some_weird_token_2", namespace="tokens")
        os.makedirs(existing_serialization_dir, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)

        self.params["vocabulary"] = {}
        self.params["vocabulary"]["type"] = "extend"
        self.params["vocabulary"]["directory"] = str(existing_vocab_path)
        self.params["vocabulary"]["min_count"] = {"tokens": 3}
        train_model(self.params, extended_serialization_dir, dry_run=True)

        vocab_files = os.listdir(extended_vocab_path)
        assert set(vocab_files) == {
            ".lock",
            "labels.txt",
            "non_padded_namespaces.txt",
            "tokens.txt",
        }

        with open(extended_vocab_path / "tokens.txt") as f:
            tokens = [line.strip() for line in f]

        assert tokens[0] == "@@UNKNOWN@@"
        assert tokens[1] == "some_weird_token_1"
        assert tokens[2] == "some_weird_token_2"

        tokens.sort()
        assert tokens == [
            ".",
            "@@UNKNOWN@@",
            "animals",
            "are",
            "some_weird_token_1",
            "some_weird_token_2",
        ]

        with open(extended_vocab_path / "labels.txt") as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == ["N", "V"]

    def test_dry_run_without_extension(self):
        existing_serialization_dir = self.TEST_DIR / "existing"
        extended_serialization_dir = self.TEST_DIR / "extended"
        existing_vocab_path = existing_serialization_dir / "vocabulary"
        extended_vocab_path = extended_serialization_dir / "vocabulary"

        vocab = Vocabulary()
        # if extend is False, its users responsibility to make sure that dataset instances
        # will be indexible by provided vocabulary. At least @@UNKNOWN@@ should be present in
        # namespace for which there could be OOV entries seen in dataset during indexing.
        # For `tokens` ns, new words will be seen but `tokens` has @@UNKNOWN@@ token.
        # but for 'labels' ns, there is no @@UNKNOWN@@ so required to add 'N', 'V' upfront.
        vocab.add_token_to_namespace("some_weird_token_1", namespace="tokens")
        vocab.add_token_to_namespace("some_weird_token_2", namespace="tokens")
        vocab.add_token_to_namespace("N", namespace="labels")
        vocab.add_token_to_namespace("V", namespace="labels")
        os.makedirs(existing_serialization_dir, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)

        self.params["vocabulary"] = {}
        self.params["vocabulary"]["type"] = "from_files"
        self.params["vocabulary"]["directory"] = str(existing_vocab_path)
        train_model(self.params, extended_serialization_dir, dry_run=True)

        with open(extended_vocab_path / "tokens.txt") as f:
            tokens = [line.strip() for line in f]

        assert tokens[0] == "@@UNKNOWN@@"
        assert tokens[1] == "some_weird_token_1"
        assert tokens[2] == "some_weird_token_2"
        assert len(tokens) == 3

    def test_make_vocab_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title="Commands", metavar="")
        Train().add_subparser(subparsers)
        for serialization_arg in ["-s", "--serialization-dir"]:
            raw_args = [
                "train",
                "path/to/params",
                serialization_arg,
                "serialization_dir",
                "--dry-run",
            ]
            args = parser.parse_args(raw_args)
            assert args.func == train_model_from_args
            assert args.param_path == "path/to/params"
            assert args.serialization_dir == "serialization_dir"
            assert args.dry_run
