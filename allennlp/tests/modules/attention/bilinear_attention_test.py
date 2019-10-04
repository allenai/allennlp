from numpy.testing import assert_almost_equal
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.modules.attention import BilinearAttention
from allennlp.common.testing import AllenNlpTestCase


class TestBilinearAttention(AllenNlpTestCase):
    def test_forward_does_a_bilinear_product(self):
        params = Params({"vector_dim": 2, "matrix_dim": 2, "normalize": False})
        bilinear = BilinearAttention.from_params(params)
        bilinear._weight_matrix = Parameter(torch.FloatTensor([[-0.3, 0.5], [2.0, -1.0]]))
        bilinear._bias = Parameter(torch.FloatTensor([0.1]))
        a_vectors = torch.FloatTensor([[1, 1]])
        b_vectors = torch.FloatTensor([[[1, 0], [0, 1]]])
        result = bilinear(a_vectors, b_vectors).detach().numpy()
        assert result.shape == (1, 2)
        assert_almost_equal(result, [[1.8, -0.4]])
