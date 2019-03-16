# pylint: disable=invalid-name,no-self-use,protected-access
import argparse
import re
import shutil

import pytest
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.fine_tune import FineTune, fine_tune_model_from_file_paths, \
                               fine_tune_model_from_args, fine_tune_model
from allennlp.common.params import Params
from allennlp.models import load_archive
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

class TestFineTune(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        self.config_file = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'experiment.json')
        self.serialization_dir = str(self.TEST_DIR / 'fine_tune')

        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title='Commands', metavar='')
        FineTune().add_subparser('fine-tune', subparsers)


    def test_fine_tune_model_runs_from_file_paths(self):
        fine_tune_model_from_file_paths(model_archive_path=self.model_archive,
                                        config_file=self.config_file,
                                        serialization_dir=self.serialization_dir)

    def test_fine_tune_does_not_expand_vocab_by_default(self):
        params = Params.from_file(self.config_file)
        # snli2 has a new token in it
        params["train_data_path"] = str(self.FIXTURES_ROOT / 'data' / 'snli2.jsonl')

        model = load_archive(self.model_archive).model

        # By default, no vocab expansion.
        fine_tune_model(model, params, self.serialization_dir)

    def test_fine_tune_works_with_vocab_expansion(self):
        params = Params.from_file(self.config_file)
        # snli2 has a new token in it
        params["train_data_path"] = str(self.FIXTURES_ROOT / 'data' / 'snli2.jsonl')

        trained_model = load_archive(self.model_archive).model
        original_weight = trained_model._text_field_embedder.token_embedder_tokens.weight

        # If we do vocab expansion, we should not get error now.
        fine_tuned_model = fine_tune_model(trained_model, params, self.serialization_dir, extend_vocab=True)
        extended_weight = fine_tuned_model._text_field_embedder.token_embedder_tokens.weight

        assert tuple(original_weight.shape) == (24, 300)
        assert tuple(extended_weight.shape) == (25, 300)
        assert torch.all(original_weight == extended_weight[:24, :])

    def test_fine_tune_works_with_vocab_expansion_with_pretrained_file(self):
        params = Params.from_file(self.config_file)
        # snli2 has a new token (seahorse) in it
        params["train_data_path"] = str(self.FIXTURES_ROOT / 'data' / 'snli2.jsonl')

        # seahorse_embeddings.gz has only token embedding for 'seahorse'.
        embeddings_filename = str(self.FIXTURES_ROOT / 'data' / 'seahorse_embeddings.gz')
        extra_token_vector = _read_pretrained_embeddings_file(embeddings_filename, 300,
                                                              Vocabulary({"tokens": {"seahorse": 1}}))[2, :]
        unavailable_embeddings_filename = "file-not-found"

        def check_embedding_extension(user_pretrained_file, saved_pretrained_file, use_pretrained):
            trained_model = load_archive(self.model_archive).model
            original_weight = trained_model._text_field_embedder.token_embedder_tokens.weight
            # Simulate the behavior of unavailable pretrained_file being stored as an attribute.
            trained_model._text_field_embedder.token_embedder_tokens._pretrained_file = saved_pretrained_file
            embedding_sources_mapping = {"_text_field_embedder.token_embedder_tokens": user_pretrained_file}
            shutil.rmtree(self.serialization_dir, ignore_errors=True)
            fine_tuned_model = fine_tune_model(trained_model, params.duplicate(),
                                               self.serialization_dir, extend_vocab=True,
                                               embedding_sources_mapping=embedding_sources_mapping)
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
        params["train_data_path"] = str(self.FIXTURES_ROOT / 'data' / 'snli2.jsonl')
        trained_model = load_archive(self.model_archive).model
        shutil.rmtree(self.serialization_dir, ignore_errors=True)
        fine_tune_model(trained_model, params.duplicate(),
                        self.serialization_dir, extend_vocab=True)
        # self.serialization_dir = str(self.TEST_DIR / 'fine_tune')
        load_archive(str(self.TEST_DIR / 'fine_tune' / "model.tar.gz"))

    def test_fine_tune_runs_from_parser_arguments(self):
        raw_args = ["fine-tune",
                    "-m", self.model_archive,
                    "-c", self.config_file,
                    "-s", self.serialization_dir]

        args = self.parser.parse_args(raw_args)

        assert args.func == fine_tune_model_from_args
        assert args.model_archive == self.model_archive
        assert args.config_file == self.config_file
        assert args.serialization_dir == self.serialization_dir
        fine_tune_model_from_args(args)

    def test_fine_tune_fails_without_required_args(self):
        # Configuration file is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args(["fine-tune", "-m", "path/to/archive", "-s", "serialization_dir"])
            assert context.exception.code == 2  # argparse code for incorrect usage

        # Serialization dir is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args(["fine-tune", "-m", "path/to/archive", "-c", "path/to/config"])
            assert context.exception.code == 2  # argparse code for incorrect usage

        # Model archive is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args(["fine-tune", "-s", "serialization_dir", "-c", "path/to/config"])
            assert context.exception.code == 2  # argparse code for incorrect usage

    def test_fine_tune_nograd_regex(self):
        original_model = load_archive(self.model_archive).model
        name_parameters_original = dict(original_model.named_parameters())
        regex_lists = [[],
                       [".*attend_feedforward.*", ".*token_embedder.*"],
                       [".*compare_feedforward.*"]]
        for regex_list in regex_lists:
            params = Params.from_file(self.config_file)
            params["trainer"]["no_grad"] = regex_list
            shutil.rmtree(self.serialization_dir, ignore_errors=True)
            tuned_model = fine_tune_model(model=original_model,
                                          params=params,
                                          serialization_dir=self.serialization_dir)
            # If regex is matched, parameter name should have requires_grad False
            # If regex is matched, parameter name should have same requires_grad
            # as the originally loaded model
            for name, parameter in tuned_model.named_parameters():
                if any(re.search(regex, name) for regex in regex_list):
                    assert not parameter.requires_grad
                else:
                    assert parameter.requires_grad \
                    == name_parameters_original[name].requires_grad
        # If all parameters have requires_grad=False, then error.
        with pytest.raises(Exception) as _:
            params = Params.from_file(self.config_file)
            params["trainer"]["no_grad"] = ["*"]
            shutil.rmtree(self.serialization_dir, ignore_errors=True)
            tuned_model = fine_tune_model(model=original_model,
                                          params=params,
                                          serialization_dir=self.serialization_dir)
