import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.attention.attention import Attention
from allennlp.modules.attention.legacy_attention import LegacyAttention


class TestLegacyAttention(AllenNlpTestCase):
    def test_can_init_legacy(self):
        legacy_attention = Attention.from_params(Params({"type": "legacy"}))
        isinstance(legacy_attention, LegacyAttention)

    def test_no_mask(self):
        attention = LegacyAttention()

        # Testing general non-batched case.
        vector = torch.FloatTensor([[0.3, 0.1, 0.5]])
        matrix = torch.FloatTensor([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]]])

        result = attention(vector, matrix).data.numpy()
        assert_almost_equal(result, numpy.array([[0.52871835, 0.47128162]]))

        # Testing non-batched case where inputs are all 0s.
        vector = torch.FloatTensor([[0, 0, 0]])
        matrix = torch.FloatTensor([[[0, 0, 0], [0, 0, 0]]])

        result = attention(vector, matrix).data.numpy()
        assert_almost_equal(result, numpy.array([[0.5, 0.5]]))

    def test_masked(self):
        attention = LegacyAttention()
        # Testing general masked non-batched case.
        vector = torch.FloatTensor([[0.3, 0.1, 0.5]])
        matrix = torch.FloatTensor([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.1, 0.4, 0.3]]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0]])
        result = attention(vector, matrix, mask).data.numpy()
        assert_almost_equal(result, numpy.array([[0.52248482, 0.0, 0.47751518]]))

    def test_batched_no_mask(self):
        attention = LegacyAttention()

        # Testing general batched case.
        vector = torch.FloatTensor([[0.3, 0.1, 0.5], [0.3, 0.1, 0.5]])
        matrix = torch.FloatTensor(
            [[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]], [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]]]
        )

        result = attention(vector, matrix).data.numpy()
        assert_almost_equal(
            result, numpy.array([[0.52871835, 0.47128162], [0.52871835, 0.47128162]])
        )

    def test_batched_masked(self):
        attention = LegacyAttention()

        # Testing general masked non-batched case.
        vector = torch.FloatTensor([[0.3, 0.1, 0.5], [0.3, 0.1, 0.5]])
        matrix = torch.FloatTensor(
            [
                [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
                [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
            ]
        )
        mask = torch.FloatTensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        result = attention(vector, matrix, mask).data.numpy()
        assert_almost_equal(
            result, numpy.array([[0.52871835, 0.47128162, 0.0], [0.50749944, 0.0, 0.49250056]])
        )

        # Test the case where a mask is all 0s and an input is all 0s.
        vector = torch.FloatTensor([[0.0, 0.0, 0.0], [0.3, 0.1, 0.5]])
        matrix = torch.FloatTensor(
            [
                [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
                [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
            ]
        )
        mask = torch.FloatTensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        result = attention(vector, matrix, mask).data.numpy()
        assert_almost_equal(result, numpy.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]))

    def test_non_normalized_attention_works(self):
        attention = LegacyAttention(normalize=False)
        sentence_tensor = torch.FloatTensor([[[-1, 0, 4], [1, 1, 1], [-1, 0, 4], [-1, 0, -1]]])
        query_tensor = torch.FloatTensor([[0.1, 0.8, 0.5]])
        result = attention(query_tensor, sentence_tensor).data.numpy()
        assert_almost_equal(result, [[1.9, 1.4, 1.9, -0.6]])

    def test_can_build_from_params(self):
        params = Params({"similarity_function": {"type": "cosine"}, "normalize": False})
        attention = LegacyAttention.from_params(params)

        assert attention._similarity_function.__class__.__name__ == "CosineSimilarity"
        assert attention._normalize is False
