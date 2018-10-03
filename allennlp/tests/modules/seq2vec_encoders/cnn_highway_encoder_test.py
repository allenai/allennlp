# pylint: disable=no-self-use
import numpy as np
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2vec_encoders.cnn_highway_encoder import CnnHighwayEncoder
from allennlp.modules.time_distributed import TimeDistributed

class TestCnnHighwayEncoder(AllenNlpTestCase):
    def test_cnn_highway_encoder(self):
        encoder = CnnHighwayEncoder(
                activation='relu',
                embedding_dim=4,
                filters=[[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                num_highway=2,
                projection_dim=16,
                projection_location='after_cnn'
        )

        encoder = TimeDistributed(encoder)

        embedding = torch.from_numpy(np.random.randn(5, 6, 50, 4)).float()
        mask = torch.ones(5, 6, 50).long()
        token_embedding = encoder(embedding, mask)

        assert list(token_embedding.size()) == [5, 6, 16]
