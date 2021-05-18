import torch
import pytest

from allennlp.common import Params
from allennlp.modules.transformer import BiModalAttention


@pytest.fixture
def params_dict():
    return {
        "hidden_size1": 6,
        "hidden_size2": 4,
        "combined_hidden_size": 16,
        "num_attention_heads": 2,
        "dropout1": 0.1,
        "dropout2": 0.2,
    }


@pytest.fixture
def params(params_dict):
    return Params(params_dict)


@pytest.fixture
def biattention(params):
    return BiModalAttention.from_params(params.duplicate())


def test_can_construct_from_params(biattention, params_dict):
    assert biattention.num_attention_heads == params_dict["num_attention_heads"]
    assert biattention.attention_head_size == int(
        params_dict["combined_hidden_size"] / params_dict["num_attention_heads"]
    )
    assert (
        biattention.all_head_size
        == params_dict["num_attention_heads"] * biattention.attention_head_size
    )
    assert biattention.query1.in_features == params_dict["hidden_size1"]
    assert biattention.key1.in_features == params_dict["hidden_size1"]
    assert biattention.value1.in_features == params_dict["hidden_size1"]
    assert biattention.dropout1.p == params_dict["dropout1"]

    assert biattention.query2.in_features == params_dict["hidden_size2"]
    assert biattention.key2.in_features == params_dict["hidden_size2"]
    assert biattention.value2.in_features == params_dict["hidden_size2"]
    assert biattention.dropout2.p == params_dict["dropout2"]


def test_forward_runs(biattention):
    biattention(
        torch.randn(2, 3, 6),
        torch.randn(2, 3, 4),
        torch.randint(0, 2, (2, 2, 3, 3)) == 1,  # creating boolean tensors
        torch.randint(0, 2, (2, 2, 3, 3)) == 1,
    )
