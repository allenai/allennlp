from numpy.testing import assert_almost_equal
import numpy
import torch
from torch.autograd import Variable
from torch.nn import Parameter

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.attention import LinearAttention
from allennlp.modules.attention.attention import Attention


class LinearAttentionTest(AllenNlpTestCase):
    def test_can_init_linear(self):
        legacy_attention = Attention.from_params(
            Params({"type": "linear", "tensor_1_dim": 3, "tensor_2_dim": 3})
        )
        isinstance(legacy_attention, LinearAttention)

    def test_linear_similarity(self):
        linear = LinearAttention(3, 3, normalize=True)
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5, 2.0, -1.0, 1, 1]))
        linear._bias = Parameter(torch.FloatTensor([0.1]))
        output = linear(
            Variable(torch.FloatTensor([[-7, -8, -9]])),
            Variable(torch.FloatTensor([[[1, 2, 3], [4, 5, 6]]])),
        )

        assert_almost_equal(output.data.numpy(), numpy.array([[0.0474, 0.9526]]), decimal=2)

    def test_bidaf_trilinear_similarity(self):
        linear = LinearAttention(2, 2, combination="x,y,x*y", normalize=False)
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5, 2.0, -1.0, 1, 1]))
        linear._bias = Parameter(torch.FloatTensor([0.0]))
        output = linear(
            torch.FloatTensor([[4, 5]]), torch.FloatTensor([[[1, 2], [4, 5], [7, 8], [10, 11]]])
        )

        assert_almost_equal(
            output.data.numpy(),
            numpy.array(
                [
                    [
                        -1.2 + 2.5 + 2 + -2 + 4 + 10,
                        -1.2 + 2.5 + 8 + -5 + 16 + 25,
                        -1.2 + 2.5 + 14 + -8 + 28 + 40,
                        -1.2 + 2.5 + 20 + -11 + 40 + 55,
                    ]
                ]
            ),
            decimal=2,
        )
