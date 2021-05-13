import torch
from torch.testing import assert_allclose
from transformers import AutoModel
import pytest

from allennlp.common import Params
from allennlp.modules.transformer import BiModalEncoder


@pytest.fixture
def params_dict():
    return {
        "num_hidden_layers1": 3,
        "num_hidden_layers2": 3,
        "hidden_size1": 12,
        "hidden_size2": 12,
        "combined_hidden_size": 12,
        "intermediate_size1": 3,
        "intermediate_size2": 3,
        "num_attention_heads1": 4,
        "num_attention_heads2": 6,
        "combined_num_attention_heads": 2,
        "attention_dropout1": 0.1,
        "hidden_dropout1": 0.2,
        "attention_dropout2": 0.1,
        "hidden_dropout2": 0.2,
        "activation": "relu",
        "biattention_id1": [1, 2],
        "biattention_id2": [1, 2],
        "fixed_layer1": 1,
        "fixed_layer2": 1,
    }


@pytest.fixture
def params(params_dict):
    return Params(params_dict)


@pytest.fixture
def bimodal_encoder(params):
    return BiModalEncoder.from_params(params.duplicate())


def test_can_construct_from_params(bimodal_encoder, params_dict):
    modules = dict(bimodal_encoder.named_modules())
    assert len(modules["layers1"]) == params_dict["num_hidden_layers1"]
    assert len(modules["layers2"]) == params_dict["num_hidden_layers2"]


def test_forward_runs(bimodal_encoder, params_dict):
    embedding1 = torch.randn(16, 34, params_dict["hidden_size1"])
    embedding2 = torch.randn(16, 2, params_dict["hidden_size2"])
    attn_mask1 = torch.randint(0, 2, (16, 1, 1, 34)) == 1
    attn_mask2 = torch.randint(0, 2, (16, 1, 1, 2)) == 1
    bimodal_encoder(embedding1, embedding2, attn_mask1, attn_mask2)


def test_loading_from_pretrained_weights(params_dict):
    pretrained_module = AutoModel.from_pretrained("bert-base-cased").encoder

    required_kwargs = [
        "num_hidden_layers2",
        "hidden_size2",
        "combined_hidden_size",
        "intermediate_size2",
        "num_attention_heads2",
        "combined_num_attention_heads",
        "attention_dropout2",
        "hidden_dropout2",
        "biattention_id1",
        "biattention_id2",
        "fixed_layer1",
        "fixed_layer2",
    ]
    kwargs = {key: params_dict[key] for key in required_kwargs}

    module = BiModalEncoder.from_pretrained_module("bert-base-cased", **kwargs)
    assert_allclose(
        module.layers1[0].intermediate.dense.weight.data,
        pretrained_module.layer[0].intermediate.dense.weight.data,
    )


def test_default_parameters():
    encoder = BiModalEncoder()
    embedding1 = torch.randn(16, 34, 1024)
    embedding2 = torch.randn(16, 2, 1024)
    attn_mask1 = torch.randint(0, 2, (16, 1, 1, 34)) == 1
    attn_mask2 = torch.randint(0, 2, (16, 1, 1, 2)) == 1

    encoder(embedding1, embedding2, attn_mask1, attn_mask2)
