# pylint: disable=invalid-name
import os
import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import train_model, _CONFIG_FILE_KEY
from allennlp.models.model import Model

import torch

class ModelTest(AllenNlpTestCase):
    def test_archiving(self):
        super(ModelTest, self).setUp()

        print(self.TEST_DIR)

        params = {
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
                "optimizer": "adam",
                "trainer": {
                        "num_epochs": 2,
                        "serialization_prefix": self.TEST_DIR
                }
        }

        # write out config file
        config_file = os.path.join(self.TEST_DIR, "config.json")
        with open(config_file, 'w') as outfile:
            outfile.write(json.dumps(params))

        params[_CONFIG_FILE_KEY] = config_file

        # `train_model` should create an archive
        model = train_model(params)

        archive_path = os.path.join(self.TEST_DIR, "model.tar.gz")

        # load from the archive
        model2 = Model.from_archive(archive_path)

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
