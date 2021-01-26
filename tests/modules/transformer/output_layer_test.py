import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import OutputLayer
from allennlp.common.testing import AllenNlpTestCase


class TestOutputLayer(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "input_size": 3,
            "hidden_size": 5,
            "dropout": 0.1,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.output_layer = OutputLayer.from_params(params)

    def test_can_construct_from_params(self):

        output_layer = self.output_layer

        assert output_layer.dense.in_features == self.params_dict["input_size"]
        assert output_layer.dense.out_features == self.params_dict["hidden_size"]

        assert output_layer.layer_norm.normalized_shape[0] == self.params_dict["hidden_size"]

        assert output_layer.dropout.p == self.params_dict["dropout"]

    def test_forward_runs(self):

        self.output_layer.forward(torch.randn(3, 3), torch.randn(3, 5))
