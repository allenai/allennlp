import numpy
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2vec_encoders import BertPooler


class TestBertPooler(AllenNlpTestCase):
    def test_encoder(self):
        encoder = BertPooler("bert-base-uncased")
        assert encoder.get_input_dim() == encoder.get_output_dim()
        embedding = torch.rand(8, 24, encoder.get_input_dim())

        pooled1 = encoder(embedding)
        assert pooled1.size() == (8, encoder.get_input_dim())

        embedding[:, 1:, :] = 0
        pooled2 = encoder(embedding)
        numpy.testing.assert_array_almost_equal(pooled1.detach().numpy(), pooled2.detach().numpy())
