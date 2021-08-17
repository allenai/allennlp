import torch
from numpy.testing import assert_almost_equal
import numpy

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.attention.attention import Attention
from allennlp.modules.attention.scaled_dot_product_attention import ScaledDotProductAttention


class TestScaledDotProductAttention(AllenNlpTestCase):
    def test_can_init_scaled_dot(self):
        legacy_attention = Attention.from_params(
            Params({"type": "scaled_dot_product", "scaling_factor": 9})
        )
        isinstance(legacy_attention, ScaledDotProductAttention)

    def test_scaled_dot_product_similarity(self):
        attn = ScaledDotProductAttention(9, normalize=False)
        vector = torch.FloatTensor([[0, 0, 0], [1, 1, 1]])
        matrix = torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        output = attn(vector, matrix)

        assert_almost_equal(output.numpy(), numpy.array([[0.0, 0.0], [8.0, 11.0]]), decimal=2)
