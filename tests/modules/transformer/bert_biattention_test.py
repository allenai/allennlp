import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import BertBiAttention
from allennlp.common.testing import AllenNlpTestCase


class TestBertBiAttention(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size1": 6,
            "hidden_size2": 4,
            "combined_hidden_size": 16,
            "num_attention_heads": 2,
            "dropout1": 0.1,
            "dropout2": 0.2,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.bert_biattention = BertBiAttention.from_params(params)

    def test_can_construct_from_params(self):

        bert_biattention = self.bert_biattention

        assert bert_biattention.num_attention_heads == self.params_dict["num_attention_heads"]
        assert bert_biattention.attention_head_size == int(
            self.params_dict["combined_hidden_size"] / self.params_dict["num_attention_heads"]
        )
        assert (
            bert_biattention.all_head_size
            == self.params_dict["num_attention_heads"] * bert_biattention.attention_head_size
        )
        assert bert_biattention.query1.in_features == self.params_dict["hidden_size1"]
        assert bert_biattention.key1.in_features == self.params_dict["hidden_size1"]
        assert bert_biattention.value1.in_features == self.params_dict["hidden_size1"]
        assert bert_biattention.dropout1.p == self.params_dict["dropout1"]

        assert bert_biattention.query2.in_features == self.params_dict["hidden_size2"]
        assert bert_biattention.key2.in_features == self.params_dict["hidden_size2"]
        assert bert_biattention.value2.in_features == self.params_dict["hidden_size2"]
        assert bert_biattention.dropout2.p == self.params_dict["dropout2"]

    def test_forward_runs(self):

        self.bert_biattention.forward(
            torch.randn(2, 3, 6),
            torch.randn(2, 2, 3, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 2, 3, 3),
        )
