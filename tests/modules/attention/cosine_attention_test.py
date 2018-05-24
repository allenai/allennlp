import torch
from numpy.testing import assert_almost_equal
import numpy as np
from torch.autograd import Variable

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.attention.cosine_attention import CosineAttention
from allennlp.modules.attention.dot_product_attention import DotProductAttention


class CosineAttentionTests(AllenNlpTestCase):

    def test_dot_product_similarity(self):
        linear = CosineAttention(False)
        output = linear(Variable(torch.FloatTensor([[0, 0, 0], [1, 1, 1]])),
                        Variable(torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])))

        assert_almost_equal(output.data.numpy(), np.array([[0.0, 0.0], [0.9948, 0.9973]]), decimal=2)



