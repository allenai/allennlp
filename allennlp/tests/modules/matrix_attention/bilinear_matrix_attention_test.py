from numpy.testing import assert_almost_equal
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.common.testing import AllenNlpTestCase


class TestBilinearMatrixAttention(AllenNlpTestCase):
    def test_forward_does_a_bilinear_product(self):
        params = Params({"matrix_1_dim": 2, "matrix_2_dim": 2})
        bilinear = BilinearMatrixAttention.from_params(params)
        bilinear._weight_matrix = Parameter(torch.FloatTensor([[-0.3, 0.5], [2.0, -1.0]]))
        bilinear._bias = Parameter(torch.FloatTensor([0.1]))
        a_vectors = torch.FloatTensor([[[1, 1], [2, 2]]])
        b_vectors = torch.FloatTensor([[[1, 0], [0, 1]]])
        result = bilinear(a_vectors, b_vectors).detach().numpy()
        assert result.shape == (1, 2, 2)
        assert_almost_equal(result, [[[1.8, -0.4], [3.5, -0.9]]])

    def test_forward_does_a_bilinear_product_when_using_biases(self):
        params = Params({"matrix_1_dim": 2, "matrix_2_dim": 2, "use_input_biases": True})
        bilinear = BilinearMatrixAttention.from_params(params)
        bilinear._weight_matrix = Parameter(
            torch.FloatTensor([[-0.3, 0.5, 1.0], [2.0, -1.0, -1.0], [1.0, 0.5, 1.0]])
        )
        bilinear._bias = Parameter(torch.FloatTensor([0.1]))
        a_vectors = torch.FloatTensor([[[1, 1], [2, 2]]])
        b_vectors = torch.FloatTensor([[[1, 0], [0, 1]]])
        result = bilinear(a_vectors, b_vectors).detach().numpy()
        assert result.shape == (1, 2, 2)
        assert_almost_equal(result, [[[3.8, 1.1], [5.5, 0.6]]])
