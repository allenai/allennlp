import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import BertSelfOutput
from allennlp.common.testing import AllenNlpTestCase


class TestBertSelfOutput(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 5,
            "dropout": 0.1,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.bert_self_output = BertSelfOutput.from_params(params)

    def test_can_construct_from_params(self):

        bert_self_output = self.bert_self_output

        assert bert_self_output.dense.in_features == self.params_dict["hidden_size"]
        assert bert_self_output.dense.out_features == self.params_dict["hidden_size"]

        assert bert_self_output.layer_norm.normalized_shape[0] == self.params_dict["hidden_size"]

        assert bert_self_output.dropout.p == self.params_dict["dropout"]

    def test_forward_runs(self):

        self.bert_self_output.forward(torch.randn(5, 5), torch.randn(5, 5))
