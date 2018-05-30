# pylint: disable=no-self-use,invalid-name

from numpy.testing import assert_allclose
import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity


class TestLegacyMatrixAttention(AllenNlpTestCase):

    def test_forward_works_on_simple_input(self):
        attention = LegacyMatrixAttention(DotProductSimilarity())
        sentence_1_tensor = torch.FloatTensor([[[1, 1, 1], [-1, 0, 1]]])
        sentence_2_tensor = torch.FloatTensor([[[1, 1, 1], [-1, 0, 1], [-1, -1, -1]]])
        result = attention(sentence_1_tensor, sentence_2_tensor).data.numpy()
        assert result.shape == (1, 2, 3)
        assert_allclose(result, [[[3, 0, -3], [0, 2, 0]]])

    def test_can_build_from_params(self):
        params = Params({"type": "legacy", 'similarity_function': {'type': 'cosine'}})
        attention = MatrixAttention.from_params(params)
        # pylint: disable=protected-access
        assert attention._similarity_function.__class__.__name__ == 'CosineSimilarity'
