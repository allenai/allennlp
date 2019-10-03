import torch
from numpy.testing import assert_almost_equal
import numpy

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.attention.attention import Attention
from allennlp.modules.attention.cosine_attention import CosineAttention


class TestCosineAttention(AllenNlpTestCase):
    def test_can_init_cosine(self):
        legacy_attention = Attention.from_params(Params({"type": "cosine"}))
        isinstance(legacy_attention, CosineAttention)

    def test_cosine_similarity(self):
        linear = CosineAttention(normalize=False)
        output = linear(
            torch.FloatTensor([[0, 0, 0], [1, 1, 1]]),
            torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
        )

        assert_almost_equal(output.numpy(), numpy.array([[0.0, 0.0], [0.9948, 0.9973]]), decimal=2)
