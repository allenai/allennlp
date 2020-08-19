import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import BertAttention
from allennlp.common.testing import AllenNlpTestCase


class TestBertAttention(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 6,
            "num_attention_heads": 2,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.2,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.bert_attention = BertAttention.from_params(params)

    def test_can_construct_from_params(self):

        bert_attention = self.bert_attention

        assert bert_attention.self.num_attention_heads == self.params_dict["num_attention_heads"]
        assert bert_attention.self.attention_head_size == int(
            self.params_dict["hidden_size"] / self.params_dict["num_attention_heads"]
        )
        assert (
            bert_attention.self.all_head_size
            == self.params_dict["num_attention_heads"] * bert_attention.self.attention_head_size
        )
        assert bert_attention.self.query.in_features == self.params_dict["hidden_size"]
        assert bert_attention.self.key.in_features == self.params_dict["hidden_size"]
        assert bert_attention.self.value.in_features == self.params_dict["hidden_size"]
        assert bert_attention.self.dropout.p == self.params_dict["attention_dropout"]

        assert bert_attention.output.dense.in_features == self.params_dict["hidden_size"]
        assert bert_attention.output.dense.out_features == self.params_dict["hidden_size"]
        assert (
            bert_attention.output.layer_norm.normalized_shape[0] == self.params_dict["hidden_size"]
        )
        assert bert_attention.output.dropout.p == self.params_dict["hidden_dropout"]

    def test_forward_runs(self):

        self.bert_attention.forward(torch.randn(2, 3, 6), torch.randn(2, 2, 3, 3))
