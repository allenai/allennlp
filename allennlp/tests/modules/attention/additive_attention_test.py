# pylint: disable=no-self-use,invalid-name,protected-access
from numpy.testing import assert_almost_equal
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.modules.attention import AdditiveAttention
from allennlp.common.testing import AllenNlpTestCase


class TestAdditiveAttention(AllenNlpTestCase):
    def test_forward_does_an_additive_product(self):
        params = Params({
                'vector_dim': 2,
                'matrix_dim': 3,
                'normalize': False,
                })
        additive = AdditiveAttention.from_params(params)
        additive._W_matrix = Parameter(torch.Tensor([[-0.2, 0.3], [-0.5, 0.5]]))
        additive._U_matrix = Parameter(torch.Tensor([[0., 1.], [1., 1.], [1., -1.]]))
        additive._V_vector = Parameter(torch.Tensor([[1.], [-1.]]))
        # batch, hidden
        vectors = torch.FloatTensor([[0.7, -0.8], [0.4, 0.9]])
        # batch, seq_len, hidden
        matrices = torch.FloatTensor([
            [[1., -1., 3.], [0.5, -0.3, 0.], [0.2, -1., 1.], [0.7, 0.8, -1.]],
            [[-2., 3., -3.], [0.6, 0.2, 2.], [0.5, -0.4, -1.], [0.2, 0.2, 0.]]])
        result = additive(vectors, matrices)
        assert result.shape == (2, 4)
        assert_almost_equal(result, [
            [[1.9751], [-0.0500], [1.2176], [-0.9206]],
            [[-1.4852], [1.4897], [-1.8903],[-1.0672]]])
