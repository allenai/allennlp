import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import SelfAttention
from allennlp.common.testing import AllenNlpTestCase


class TestSelfAttention(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 6,
            "num_attention_heads": 2,
            "dropout": 0.0,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.self_attention = SelfAttention.from_params(params)

    def test_can_construct_from_params(self):
        assert self.self_attention.num_attention_heads == self.params_dict["num_attention_heads"]
        assert self.self_attention.attention_head_size == int(
            self.params_dict["hidden_size"] / self.params_dict["num_attention_heads"]
        )

        assert (
            self.self_attention.all_head_size
            == self.params_dict["num_attention_heads"] * self.self_attention.attention_head_size
        )

        assert self.self_attention.query.in_features == self.params_dict["hidden_size"]
        assert self.self_attention.key.in_features == self.params_dict["hidden_size"]
        assert self.self_attention.value.in_features == self.params_dict["hidden_size"]

        assert self.self_attention.dropout.p == self.params_dict["dropout"]

    def test_forward_runs(self):
        self.self_attention.forward(torch.randn(2, 3, 6), torch.randn(2, 2, 3, 3))
