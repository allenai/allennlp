import argparse
import os

import pytest

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data import DataLoader
from allennlp.models import Model
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, requires_multi_gpu
from allennlp.commands.find_learning_rate import (
    search_learning_rate,
    find_learning_rate_from_args,
    find_learning_rate_model,
    FindLearningRate,
)
from allennlp.training import Trainer
from allennlp.training.util import datasets_from_params


def is_matplotlib_installed():
    try:
        import matplotlib  # noqa: F401 - Matplotlib is optional.
    except:  # noqa: E722. Any exception means we don't have a working matplotlib.
        return False
    return True


class TestFindLearningRate(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.params = lambda: Params(
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
                "trainer": {"cuda_device": -1, "num_epochs": 2, "optimizer": "adam"},
            }
        )

    @pytest.mark.skipif(not is_matplotlib_installed(), reason="matplotlib dependency is optional")
    def test_find_learning_rate(self):
        find_learning_rate_model(
            self.params(),
            os.path.join(self.TEST_DIR, "test_find_learning_rate"),
            start_lr=1e-5,
            end_lr=1,
            num_batches=100,
            linear_steps=True,
            stopping_factor=None,
            force=False,
        )

        # It's OK if serialization dir exists but is empty:
        serialization_dir2 = os.path.join(self.TEST_DIR, "empty_directory")
        assert not os.path.exists(serialization_dir2)
        os.makedirs(serialization_dir2)
        find_learning_rate_model(
            self.params(),
            serialization_dir2,
            start_lr=1e-5,
            end_lr=1,
            num_batches=100,
            linear_steps=True,
            stopping_factor=None,
            force=False,
        )

        # It's not OK if serialization dir exists and has junk in it non-empty:
        serialization_dir3 = os.path.join(self.TEST_DIR, "non_empty_directory")
        assert not os.path.exists(serialization_dir3)
        os.makedirs(serialization_dir3)
        with open(os.path.join(serialization_dir3, "README.md"), "w") as f:
            f.write("TEST")

        with pytest.raises(ConfigurationError):
            find_learning_rate_model(
                self.params(),
                serialization_dir3,
                start_lr=1e-5,
                end_lr=1,
                num_batches=100,
                linear_steps=True,
                stopping_factor=None,
                force=False,
            )

        # ... unless you use the --force flag.
        find_learning_rate_model(
            self.params(),
            serialization_dir3,
            start_lr=1e-5,
            end_lr=1,
            num_batches=100,
            linear_steps=True,
            stopping_factor=None,
            force=True,
        )

    def test_find_learning_rate_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title="Commands", metavar="")
        FindLearningRate().add_subparser(subparsers)

        for serialization_arg in ["-s", "--serialization-dir"]:
            raw_args = ["find-lr", "path/to/params", serialization_arg, "serialization_dir"]

            args = parser.parse_args(raw_args)

            assert args.func == find_learning_rate_from_args
            assert args.param_path == "path/to/params"
            assert args.serialization_dir == "serialization_dir"

        # config is required
        with pytest.raises(SystemExit) as cm:
            parser.parse_args(["find-lr", "-s", "serialization_dir"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

        # serialization dir is required
        with pytest.raises(SystemExit) as cm:
            parser.parse_args(["find-lr", "path/to/params"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

    @requires_multi_gpu
    def test_find_learning_rate_multi_gpu(self):
        params = self.params()
        del params["trainer"]["cuda_device"]
        params["distributed"] = Params({})
        params["distributed"]["cuda_devices"] = [0, 1]

        with pytest.raises(AssertionError) as execinfo:
            find_learning_rate_model(
                params,
                os.path.join(self.TEST_DIR, "test_find_learning_rate_multi_gpu"),
                start_lr=1e-5,
                end_lr=1,
                num_batches=100,
                linear_steps=True,
                stopping_factor=None,
                force=False,
            )
        assert "DistributedDataParallel" in str(execinfo.value)


class TestSearchLearningRate(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
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
                "train_data_path": str(self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"),
                "validation_data_path": str(self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"),
                "data_loader": {"batch_size": 2},
                "trainer": {"cuda_device": -1, "num_epochs": 2, "optimizer": "adam"},
            }
        )
        all_datasets = datasets_from_params(params)
        vocab = Vocabulary.from_params(
            params.pop("vocabulary", {}),
            instances=(instance for dataset in all_datasets.values() for instance in dataset),
        )
        model = Model.from_params(vocab=vocab, params=params.pop("model"))
        train_data = all_datasets["train"]
        train_data.index_with(vocab)

        data_loader = DataLoader.from_params(dataset=train_data, params=params.pop("data_loader"))
        trainer_params = params.pop("trainer")
        serialization_dir = os.path.join(self.TEST_DIR, "test_search_learning_rate")

        self.trainer = Trainer.from_params(
            model=model,
            serialization_dir=serialization_dir,
            data_loader=data_loader,
            train_data=train_data,
            params=trainer_params,
            validation_data=None,
            validation_iterator=None,
        )

    def test_search_learning_rate_with_num_batches_less_than_ten(self):
        with pytest.raises(ConfigurationError):
            search_learning_rate(self.trainer, num_batches=9)

    def test_search_learning_rate_linear_steps(self):
        learning_rates_losses = search_learning_rate(self.trainer, linear_steps=True)
        assert len(learning_rates_losses) > 1

    def test_search_learning_rate_without_stopping_factor(self):
        learning_rates, losses = search_learning_rate(
            self.trainer, num_batches=100, stopping_factor=None
        )
        assert len(learning_rates) == 101
        assert len(losses) == 101
