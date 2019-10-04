from numpy.testing import assert_almost_equal
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.modules.attention import AdditiveAttention
from allennlp.common.testing import AllenNlpTestCase


class TestAdditiveAttention(AllenNlpTestCase):
    def test_forward_does_an_additive_product(self):
        params = Params({"vector_dim": 2, "matrix_dim": 3, "normalize": False})
        additive = AdditiveAttention.from_params(params)
        additive._w_matrix = Parameter(torch.Tensor([[-0.2, 0.3], [-0.5, 0.5]]))
        additive._u_matrix = Parameter(torch.Tensor([[0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]))
        additive._v_vector = Parameter(torch.Tensor([[1.0], [-1.0]]))
        vectors = torch.FloatTensor([[0.7, -0.8], [0.4, 0.9]])
        matrices = torch.FloatTensor(
            [
                [[1.0, -1.0, 3.0], [0.5, -0.3, 0.0], [0.2, -1.0, 1.0], [0.7, 0.8, -1.0]],
                [[-2.0, 3.0, -3.0], [0.6, 0.2, 2.0], [0.5, -0.4, -1.0], [0.2, 0.2, 0.0]],
            ]
        )
        result = additive(vectors, matrices).detach().numpy()
        assert result.shape == (2, 4)
        assert_almost_equal(
            result,
            [
                [1.975072, -0.04997836, 1.2176098, -0.9205586],
                [-1.4851665, 1.489604, -1.890285, -1.0672251],
            ],
        )
