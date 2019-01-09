# pylint: disable=no-self-use,invalid-name,protected-access
from numpy.testing import assert_almost_equal
import torch
from torch.nn.parameter import Parameter
from allennlp.common import Params
from allennlp.modules.seq2vec_encoders import AttentionEncoder
from allennlp.common.testing import AllenNlpTestCase


class TestAttentionEncoder(AllenNlpTestCase):
    def test_forward(self):
        params = Params({
                "input_dim": 4,
                "context_vector_dim": 4
                })
        han_attention = AttentionEncoder.from_params(params)
        han_attention._mlp.weight = Parameter(torch.FloatTensor([[-.3, .5], [2.0, -1.0]]))
        han_attention._mlp.bias = Parameter(torch.FloatTensor([.1]))
        han_attention._context_dot_product.weight = Parameter(torch.FloatTensor([[-.3, .5]]))
        matrix = torch.FloatTensor([[[1.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [1.0, 0.0]]])
        matrix_mask = torch.FloatTensor([[1.0, 1.0], [1.0, 0.0]])
        result = han_attention(matrix, matrix_mask).detach().numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[1.0, 0.4423617], [1.0, 1.0]])
