import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import BertIntermediate
from allennlp.common.testing import AllenNlpTestCase


class TestBertIntermediate(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 5,
            "intermediate_size": 3,
            "activation": "relu",
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.bert_intermediate = BertIntermediate.from_params(params)

    def test_can_construct_from_params(self):

        bert_intermediate = self.bert_intermediate

        assert bert_intermediate.dense.in_features == self.params_dict["hidden_size"]
        assert bert_intermediate.dense.out_features == self.params_dict["intermediate_size"]

    def test_forward_runs(self):

        self.bert_intermediate.forward(torch.randn(7, 5))
