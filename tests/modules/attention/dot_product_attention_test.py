import torch
from numpy.testing import assert_almost_equal
import numpy as np
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.attention.attention import Attention
from allennlp.modules.attention.dot_product_attention import DotProductAttention


class TestDotProductAttention(AllenNlpTestCase):

    def test_can_init_dot(self):
        legacy_attention = Attention.from_params(Params({"type": "dot_product"}))
        isinstance(legacy_attention, DotProductAttention)

    def test_dot_product_similarity(self):
        linear = DotProductAttention(normalize=False)
        output = linear(Variable(torch.FloatTensor([[0, 0, 0], [1, 1, 1]])),
                        Variable(torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])))

        assert_almost_equal(output.data.numpy(),
                            np.array([[0.0, 0.0], [24.0, 33.0]]), decimal=2)



