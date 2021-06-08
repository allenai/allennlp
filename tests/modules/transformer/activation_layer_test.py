import torch
import pytest

from allennlp.common import Params
from allennlp.modules.transformer import ActivationLayer


@pytest.fixture
def params_dict():
    return {
        "hidden_size": 5,
        "intermediate_size": 3,
        "activation": "relu",
    }


@pytest.fixture
def params(params_dict):
    return Params(params_dict)


@pytest.fixture
def activation_layer(params):
    return ActivationLayer.from_params(params.duplicate())


def test_can_construct_from_params(activation_layer, params_dict):
    activation_layer = activation_layer
    assert activation_layer.dense.in_features == params_dict["hidden_size"]
    assert activation_layer.dense.out_features == params_dict["intermediate_size"]


def test_forward_runs(activation_layer):
    activation_layer.forward(torch.randn(7, 5))
