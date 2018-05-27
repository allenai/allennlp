# pylint: disable=invalid-name,no-self-use,protected-access
import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable
from torch.nn import Parameter

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.matrix_attention import LinearMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


class TestLinearMatrixAttention(AllenNlpTestCase):

    def test_can_init_dot(self):
        legacy_attention = MatrixAttention.from_params(Params({"type": "linear",
                                                               "tensor_1_dim": 3,
                                                               "tensor_2_dim": 3}))
        isinstance(legacy_attention, LinearMatrixAttention)

    def test_linear_similarity(self):
        linear = LinearMatrixAttention(3, 3)
        linear._weight_vector = Parameter(torch.FloatTensor([-.3, .5, 2.0, -1.0, 1, 1]))
        linear._bias = Parameter(torch.FloatTensor([.1]))
        output = linear(Variable(torch.FloatTensor([[[0, 0, 0], [4, 5, 6]], [[-7, -8, -9], [10, 11, 12]]])),
                        Variable(torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])))

        assert_almost_equal(output.data.numpy(), numpy.array([[[4.1000, 7.1000], [17.4000, 20.4000]],
                                                              [[-9.8000, -6.8000], [36.6000, 39.6000]]]),
                            decimal=2)
