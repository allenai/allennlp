# pylint: disable=no-self-use
import numpy as np
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.token_embedders.cnn_highway_encoder import CnnHighwayEncoder

class TestCnnHighwayEncoder(AllenNlpTestCase):
    def test_cnn_highway_encoder(self):
        encoder = CnnHighwayEncoder(
                activation='relu',
                embedding_dim=4,
                filters=[[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                max_characters_per_token=50,
                num_characters=262,
                num_highway=2,
                projection_dim=16,
                projection_location='after_cnn'
        )

        character_ids = torch.from_numpy(np.random.randint(0, 262, size=(5, 6, 50)))
        token_embedding = encoder(character_ids)

        assert list(token_embedding.size()) == [5, 6, 16]
