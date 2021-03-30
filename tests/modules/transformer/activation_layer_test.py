import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import ActivationLayer
from allennlp.common.testing import AllenNlpTestCase


class TestActivationLayer(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 5,
            "intermediate_size": 3,
            "activation": "relu",
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.activation_layer = ActivationLayer.from_params(params)

    def test_can_construct_from_params(self):

        activation_layer = self.activation_layer

        assert activation_layer.dense.in_features == self.params_dict["hidden_size"]
        assert activation_layer.dense.out_features == self.params_dict["intermediate_size"]

    def test_forward_runs(self):

        self.activation_layer.forward(torch.randn(7, 5))
