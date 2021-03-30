import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.nn import Parameter

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.matrix_attention import LinearMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


class TestLinearMatrixAttention(AllenNlpTestCase):
    def test_can_init_dot(self):
        legacy_attention = MatrixAttention.from_params(
            Params({"type": "linear", "tensor_1_dim": 3, "tensor_2_dim": 3})
        )
        isinstance(legacy_attention, LinearMatrixAttention)

    def test_linear_similarity(self):
        linear = LinearMatrixAttention(3, 3)
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5, 2.0, -1.0, 1, 1]))
        linear._bias = Parameter(torch.FloatTensor([0.1]))
        output = linear(
            torch.FloatTensor([[[0, 0, 0], [4, 5, 6]], [[-7, -8, -9], [10, 11, 12]]]),
            torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
        )

        assert_almost_equal(
            output.data.numpy(),
            numpy.array(
                [[[4.1000, 7.1000], [17.4000, 20.4000]], [[-9.8000, -6.8000], [36.6000, 39.6000]]]
            ),
            decimal=2,
        )

    def test_bidaf_trilinear_similarity(self):
        linear = LinearMatrixAttention(2, 2, combination="x,y,x*y")
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5, 2.0, -1.0, 1, 1]))
        linear._bias = Parameter(torch.FloatTensor([0.0]))
        output = linear(
            torch.FloatTensor([[[0, 0], [4, 5]], [[-7, -8], [10, 11]]]),
            torch.FloatTensor([[[1, 2], [4, 5]], [[7, 8], [10, 11]]]),
        )

        assert_almost_equal(
            output.data.numpy(),
            numpy.array(
                [
                    [
                        [0 + 0 + 2 + -2 + 0 + 0, 0 + 0 + 8 + -5 + 0 + 0],
                        [-1.2 + 2.5 + 2 + -2 + 4 + 10, -1.2 + 2.5 + 8 + -5 + 16 + 25],
                    ],
                    [
                        [2.1 + -4 + 14 + -8 + -49 + -64, 2.1 + -4 + 20 + -11 + -70 + -88],
                        [-3 + 5.5 + 14 + -8 + 70 + 88, -3 + 5.5 + 20 + -11 + 100 + 121],
                    ],
                ]
            ),
            decimal=2,
        )
