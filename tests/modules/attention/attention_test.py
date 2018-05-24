from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.attention.cosine_attention import CosineAttention
from allennlp.modules.attention.legacy_attention import Attention, LegacyAttention


class AttentionTests(AllenNlpTestCase):

    def test_can_init_legacy(self):
        legacy_attention = Attention.from_params(Params({"type": "legacy"}))
        isinstance(legacy_attention, LegacyAttention)

    def test_can_init_dot(self):
        legacy_attention = Attention.from_params(Params({"type": "dot_product"}))
        isinstance(legacy_attention, DotProductAttention)

    def test_can_init_cosine(self):
        legacy_attention = Attention.from_params(Params({"type": "cosine"}))
        isinstance(legacy_attention, CosineAttention)
