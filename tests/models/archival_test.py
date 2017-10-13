# pylint: disable=invalid-name
import os
import copy

import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import train_model
from allennlp.models.archival import load_archive


class ArchivalTest(AllenNlpTestCase):
    def test_archiving(self):
        super(ArchivalTest, self).setUp()

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
                        "optimizer": "adam",
                }
        })

        # copy params, since they'll get consumed during training
        params_copy = copy.deepcopy(params.as_dict())

        # `train_model` should create an archive
        model = train_model(params, serialization_dir=self.TEST_DIR)

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
