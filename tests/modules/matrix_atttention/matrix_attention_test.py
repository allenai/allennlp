from allennlp.common.params import Params

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.matrix_attention import CosineMatrixAttention, DotProductMatrixAttention
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


class MatrixAttentionTests(AllenNlpTestCase):

    def test_can_init_legacy(self):
        legacy_attention = MatrixAttention.from_params(Params({"type": "legacy"}))
        isinstance(legacy_attention, LegacyMatrixAttention)

    def test_can_init_dot(self):
        legacy_attention = MatrixAttention.from_params(Params({"type": "dot_product"}))
        isinstance(legacy_attention, DotProductMatrixAttention)

    def test_can_init_cosine(self):
        legacy_attention = MatrixAttention.from_params(Params({"type": "cosine"}))
        isinstance(legacy_attention, CosineMatrixAttention)
