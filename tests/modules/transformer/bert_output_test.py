import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import BertOutput
from allennlp.common.testing import AllenNlpTestCase


class TestBertOutput(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 5,
            "intermediate_size": 3,
            "dropout": 0.1,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.bert_output = BertOutput.from_params(params)

    def test_can_construct_from_params(self):

        bert_output = self.bert_output

        assert bert_output.dense.in_features == self.params_dict["intermediate_size"]
        assert bert_output.dense.out_features == self.params_dict["hidden_size"]

        assert bert_output.layer_norm.normalized_shape[0] == self.params_dict["hidden_size"]

        assert bert_output.dropout.p == self.params_dict["dropout"]

    def test_forward_runs(self):

        self.bert_output.forward(torch.randn(3, 3), torch.randn(3, 5))
