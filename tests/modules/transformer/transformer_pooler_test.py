import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import TransformerPooler
from allennlp.common.testing import AllenNlpTestCase


class TestTransformerPooler(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 5,
            "intermediate_size": 3,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.pooler = TransformerPooler.from_params(params)

    def test_can_construct_from_params(self):

        assert self.pooler.dense.in_features == self.params_dict["hidden_size"]
        assert self.pooler.dense.out_features == self.params_dict["intermediate_size"]

    def test_forward_runs(self):

        out = self.pooler.forward(torch.randn(2, 7, 5))
        assert out.size() == (2, 3)
