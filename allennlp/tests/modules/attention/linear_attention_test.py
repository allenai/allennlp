# pylint: disable=invalid-name,no-self-use,protected-access
from numpy.testing import assert_almost_equal
import numpy
import torch
from torch.autograd import Variable
from torch.nn import Parameter

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.attention import LinearAttention
from allennlp.modules.attention.attention import Attention


class LinearAttentionTests(AllenNlpTestCase):

    def test_can_init_linear(self):
        legacy_attention = Attention.from_params(Params({"type": "linear",
                                                         "tensor_1_dim": 3,
                                                         "tensor_2_dim": 3}))
        isinstance(legacy_attention, LinearAttention)

    def test_linear_similarity(self):
        linear = LinearAttention(3, 3, normalize=True)
        linear._weight_vector = Parameter(torch.FloatTensor([-.3, .5, 2.0, -1.0, 1, 1]))
        linear._bias = Parameter(torch.FloatTensor([.1]))
        output = linear(Variable(torch.FloatTensor([[-7, -8, -9]])),
                        Variable(torch.FloatTensor([[[1, 2, 3], [4, 5, 6]]])))

        assert_almost_equal(output.data.numpy(), numpy.array([[0.0474, 0.9526]]), decimal=2)
