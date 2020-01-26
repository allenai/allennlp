import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from typing import Iterable

import pytest
import torch

from allennlp.commands.train import Train, train_model, train_model_from_args
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

    def test_vocab_extended_model_with_transferred_embedder_is_loadable(self):
        # Train on snli2 but load text_field_embedder and vocab from the model
        # trained on snli (snli2 has one extra token over snli).
        # Make sure (1) embedding extension happens implicitly.
        #           (2) model dumped in such a way is loadable.
        # (1) corresponds to model.extend_embedder_vocab() in trainer.py
        # (2) corresponds to model.extend_embedder_vocab() in model.py
        config_file = str(self.FIXTURES_ROOT / "decomposable_attention" / "experiment.json")
        model_archive = str(
            self.FIXTURES_ROOT / "decomposable_attention" / "serialization" / "model.tar.gz"
        )
        serialization_dir = str(self.TEST_DIR / "train")

        params = Params.from_file(config_file).as_dict()

        snli_vocab_path = str(self.FIXTURES_ROOT / "data" / "snli_vocab")
        params["train_data_path"] = str(self.FIXTURES_ROOT / "data" / "snli2.jsonl")
        params["model"]["text_field_embedder"] = {
            "_pretrained": {"archive_file": model_archive, "module_path": "_text_field_embedder"}
        }
        params["vocabulary"] = {"type": "extend", "directory": snli_vocab_path}

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
        load_archive(str(self.TEST_DIR / "train" / "model.tar.gz"))


class TestFineTune(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.model_archive = str(
            self.FIXTURES_ROOT / "decomposable_attention" / "serialization" / "model.tar.gz"
        )
        self.config_file = str(self.FIXTURES_ROOT / "decomposable_attention" / "experiment.json")
        self.serialization_dir = str(self.TEST_DIR / "fine_tune")

    def test_fine_tune_does_not_expand_vocab_by_default(self):
        params = Params.from_file(self.config_file)
        # snli2 has a new token in it
        params["train_data_path"] = str(self.FIXTURES_ROOT / "data" / "snli2.jsonl")

        model = load_archive(self.model_archive).model

        # By default, no vocab expansion.
        train_model(params, self.serialization_dir, model=model)

    def test_fine_tune_works_with_vocab_expansion(self):
        params = Params.from_file(self.config_file)
        # snli2 has a new token in it
        params["train_data_path"] = str(self.FIXTURES_ROOT / "data" / "snli2.jsonl")

        trained_model = load_archive(self.model_archive).model
        original_weight = trained_model._text_field_embedder.token_embedder_tokens.weight

        # If we do vocab expansion, we should not get error now.
        fine_tuned_model = train_model(
            params, self.serialization_dir, model=trained_model, extend_vocab=True
        )
        extended_weight = fine_tuned_model._text_field_embedder.token_embedder_tokens.weight

        assert tuple(original_weight.shape) == (24, 300)
        assert tuple(extended_weight.shape) == (25, 300)
        assert torch.all(original_weight == extended_weight[:24, :])

    def test_fine_tune_works_with_vocab_expansion_with_pretrained_file(self):
        params = Params.from_file(self.config_file)
        # snli2 has a new token (seahorse) in it
        params["train_data_path"] = str(self.FIXTURES_ROOT / "data" / "snli2.jsonl")

        # seahorse_embeddings.gz has only token embedding for 'seahorse'.
        embeddings_filename = str(self.FIXTURES_ROOT / "data" / "seahorse_embeddings.gz")
        extra_token_vector = _read_pretrained_embeddings_file(
            embeddings_filename, 300, Vocabulary({"tokens": {"seahorse": 1}})
        )[2, :]
        unavailable_embeddings_filename = "file-not-found"

        def check_embedding_extension(user_pretrained_file, saved_pretrained_file, use_pretrained):
            trained_model = load_archive(self.model_archive).model
            original_weight = trained_model._text_field_embedder.token_embedder_tokens.weight
            # Simulate the behavior of unavailable pretrained_file being stored as an attribute.
            trained_model._text_field_embedder.token_embedder_tokens._pretrained_file = (
                saved_pretrained_file
            )
            embedding_sources_mapping = {
                "_text_field_embedder.token_embedder_tokens": user_pretrained_file
            }
            shutil.rmtree(self.serialization_dir, ignore_errors=True)
            fine_tuned_model = train_model(
                params.duplicate(),
                self.serialization_dir,
                model=trained_model,
                extend_vocab=True,
                embedding_sources_mapping=embedding_sources_mapping,
            )
            extended_weight = fine_tuned_model._text_field_embedder.token_embedder_tokens.weight
            assert original_weight.shape[0] + 1 == extended_weight.shape[0] == 25
            assert torch.all(original_weight == extended_weight[:24, :])
            if use_pretrained:
                assert torch.all(extended_weight[24, :] == extra_token_vector)
            else:
                assert torch.all(extended_weight[24, :] != extra_token_vector)

        # TEST 1: Passing correct embedding_sources_mapping should work when pretrained_file attribute
        #         wasn't stored. (Model archive was generated without behaviour of storing pretrained_file)
        check_embedding_extension(embeddings_filename, None, True)

        # TEST 2: Passing correct embedding_sources_mapping should work when pretrained_file
        #         attribute was stored and user's choice should take precedence.
        check_embedding_extension(embeddings_filename, unavailable_embeddings_filename, True)

        # TEST 3: Passing no embedding_sources_mapping should work, if available pretrained_file
        #         attribute was stored.
        check_embedding_extension(None, embeddings_filename, True)

        # TEST 4: Passing incorrect pretrained-file by mapping should raise error.
        with pytest.raises(ConfigurationError):
            check_embedding_extension(unavailable_embeddings_filename, embeddings_filename, True)

        # TEST 5: If none is available, it should NOT raise error. Pretrained file could
        #         possibly not have been used in first place.
        check_embedding_extension(None, unavailable_embeddings_filename, False)

    def test_fine_tune_extended_model_is_loadable(self):
        params = Params.from_file(self.config_file)
        # snli2 has a new token (seahorse) in it
        params["train_data_path"] = str(self.FIXTURES_ROOT / "data" / "snli2.jsonl")
        trained_model = load_archive(self.model_archive).model
        shutil.rmtree(self.serialization_dir, ignore_errors=True)
        train_model(
            params.duplicate(), self.serialization_dir, model=trained_model, extend_vocab=True
        )
        # self.serialization_dir = str(self.TEST_DIR / 'fine_tune')
        load_archive(str(self.TEST_DIR / "fine_tune" / "model.tar.gz"))

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
