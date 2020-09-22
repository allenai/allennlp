import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer.attention_scores import (
    GeneralAttention,
    AdditiveAttention,
    DotProduct,
    ScaledDotProduct,
    ContentBaseAttention,
)
from allennlp.common.testing import AllenNlpTestCase


class TestGeneralAttention(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 3,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.general_attention = GeneralAttention.from_params(params)

    def test_can_construct_from_params(self):
        assert self.general_attention.hidden_size == self.params_dict["hidden_size"]

    def test_forward_runs(self):
        self.general_attention(torch.randn(4, 3), torch.randn(4, 3))


class TestAdditiveAttention(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 3,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.additive_attention = AdditiveAttention.from_params(params)

    def test_can_construct_from_params(self):
        assert self.additive_attention.Wa.in_features == 2 * self.params_dict["hidden_size"]
        assert self.additive_attention.Wa.out_features == self.params_dict["hidden_size"]
        assert self.additive_attention.va.in_features == self.params_dict["hidden_size"]

    def test_forward_runs(self):
        self.additive_attention(torch.randn(4, 3), torch.randn(4, 3))


class TestDotProduct(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.dot_product = DotProduct()

    def test_forward_runs(self):
        self.dot_product(torch.randn(4, 3), torch.randn(4, 3))


class TestScaledDotProduct(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.scaled_dot_product = ScaledDotProduct(8)

    def test_forward_runs(self):
        self.scaled_dot_product(torch.randn(4, 3), torch.randn(4, 3))


class TestContentBaseAttention(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.content_base = ContentBaseAttention()

    def test_forward_runs(self):
        self.content_base(torch.randn(4, 3), torch.randn(4, 3))
