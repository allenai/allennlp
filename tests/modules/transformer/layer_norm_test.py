import copy
import torch
import pytest

from allennlp.common import Params
from allennlp.modules.transformer import LayerNorm
from allennlp.common.testing import AllenNlpTestCase

PARAMS_DICT = {
    "normalized_shape": (2, 3),
    "eps": 1e-05,
    "elementwise_affine": True,
}


@pytest.fixture
def params_dict():
    return copy.deepcopy(PARAMS_DICT)


@pytest.fixture
def layer_norm(params_dict):
    return LayerNorm(**params_dict)


def test_can_construct_from_params(layer_norm, params_dict):
    assert layer_norm.normalized_shape == params_dict["normalized_shape"]
    assert layer_norm.eps == params_dict["eps"]
    assert layer_norm.elementwise_affine == params_dict["elementwise_affine"]


def test_forward_runs(layer_norm):
    inputs = torch.randn(4, 2, 3)
    output = layer_norm.forward(inputs)
    scripted = torch.jit.script(layer_norm)
    scripted_output = scripted.forward(inputs)
    assert torch.allclose(output, scripted_output)
