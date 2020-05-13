import copy

import torch

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import archive_model, load_archive


def assert_models_equal(model, model2):
    # check that model weights are the same
    keys = set(model.state_dict().keys())
    keys2 = set(model2.state_dict().keys())

    assert keys == keys2

    for key in keys:
        assert torch.equal(model.state_dict()[key], model2.state_dict()[key])

    # check that vocabularies are the same
    vocab = model.vocab
    vocab2 = model2.vocab

    assert vocab._token_to_index == vocab2._token_to_index
    assert vocab._index_to_token == vocab2._index_to_token


class ArchivalTest(AllenNlpTestCase):
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

    def test_archiving(self):
        # copy params, since they'll get consumed during training
        params_copy = copy.deepcopy(self.params.as_dict())

        # `train_model` should create an archive
        serialization_dir = self.TEST_DIR / "archive_test"
        model = train_model(self.params, serialization_dir=serialization_dir)

        archive_path = serialization_dir / "model.tar.gz"

        # load from the archive
        archive = load_archive(archive_path)
        model2 = archive.model

        assert_models_equal(model, model2)

        # check that params are the same
        params2 = archive.config
        assert params2.as_dict() == params_copy

    def test_archive_model_uses_archive_path(self):

        serialization_dir = self.TEST_DIR / "serialization"
        # Train a model
        train_model(self.params, serialization_dir=serialization_dir)
        # Use a new path.
        archive_model(
            serialization_dir=serialization_dir, archive_path=serialization_dir / "new_path.tar.gz"
        )
        archive = load_archive(serialization_dir / "new_path.tar.gz")
        assert archive

    def test_loading_serialization_directory(self):
        # copy params, since they'll get consumed during training
        params_copy = copy.deepcopy(self.params.as_dict())

        # `train_model` should create an archive
        serialization_dir = self.TEST_DIR / "serialization"
        model = train_model(self.params, serialization_dir=serialization_dir)

        # load from the serialization directory itself
        archive = load_archive(serialization_dir)
        model2 = archive.model

        assert_models_equal(model, model2)

        # check that params are the same
        params2 = archive.config
        assert params2.as_dict() == params_copy

    def test_can_load_from_archive_model(self):
        serialization_dir = self.FIXTURES_ROOT / "basic_classifier" / "from_archive_serialization"
        archive_path = serialization_dir / "model.tar.gz"
        model = load_archive(archive_path).model

        # We want to be sure that we don't just not crash, but also be sure that we loaded the right
        # weights for the model.  We'll do that by making sure that we didn't just load the model
        # that's in the `archive_path` of the config file, which is this one.
        base_model_path = self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        base_model = load_archive(base_model_path).model
        base_model_params = dict(base_model.named_parameters())
        for name, parameters in model.named_parameters():
            if parameters.size() == base_model_params[name].size():
                assert not (parameters == base_model_params[name]).all()
            else:
                # In this case, the parameters are definitely different, no need for the above
                # check.
                pass
