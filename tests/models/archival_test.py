# pylint: disable=invalid-name
import os
import copy

import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import train_model
from allennlp.models.archival import load_archive, archive_model


class ArchivalTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.params = Params({
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
                        "optimizer": "adam",
                }
        })

    def test_archiving(self):
        # copy params, since they'll get consumed during training
        params_copy = copy.deepcopy(self.params.as_dict())

        # `train_model` should create an archive
        model = train_model(self.params, serialization_dir=self.TEST_DIR)

        archive_path = os.path.join(self.TEST_DIR, "model.tar.gz")

        # load from the archive
        archive = load_archive(archive_path)
        model2 = archive.model

        # check that model weights are the same
        keys = set(model.state_dict().keys())
        keys2 = set(model2.state_dict().keys())

        assert keys == keys2

        for key in keys:
            assert torch.equal(model.state_dict()[key], model2.state_dict()[key])

        # check that vocabularies are the same
        vocab = model.vocab
        vocab2 = model2.vocab

        assert vocab._token_to_index == vocab2._token_to_index  # pylint: disable=protected-access
        assert vocab._index_to_token == vocab2._index_to_token  # pylint: disable=protected-access

        # check that params are the same
        params2 = archive.config
        assert params2.as_dict() == params_copy

    def test_extra_files(self):

        serialization_dir = os.path.join(self.TEST_DIR, 'serialization')

        # Train a model
        train_model(self.params, serialization_dir=serialization_dir)

        # Archive model, and also archive the training data
        files_to_archive = {"train_data_path": 'tests/fixtures/data/sequence_tagging.tsv'}
        archive_model(serialization_dir=serialization_dir, files_to_archive=files_to_archive)

        archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'))
        params = archive.config

        # The param in the data should have been replaced with a temporary path
        # (which we don't know, but we know what it ends with).
        assert params.get('train_data_path').endswith('/fta/train_data_path')

        # The validation data path should be the same though.
        assert params.get('validation_data_path') == 'tests/fixtures/data/sequence_tagging.tsv'
