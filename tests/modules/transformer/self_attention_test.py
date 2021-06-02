import copy

import torch
import pytest
from transformers import AutoModel

from allennlp.common import Params

from allennlp.modules.transformer.attention_module import SelfAttention
from allennlp.nn.util import min_value_of_dtype


PARAMS_DICT = {
    "hidden_size": 6,
    "num_attention_heads": 2,
    "dropout": 0.0,
}


@pytest.fixture
def params_dict():
    return copy.deepcopy(PARAMS_DICT)


@pytest.fixture
def params(params_dict):
    return Params(params_dict)


@pytest.fixture
def self_attention(params):
    return SelfAttention.from_params(params.duplicate())


def test_can_construct_from_params(self_attention, params_dict):
    assert self_attention.num_attention_heads == params_dict["num_attention_heads"]
    assert self_attention.attention_head_size == int(
        params_dict["hidden_size"] / params_dict["num_attention_heads"]
    )

    assert (
        self_attention.all_head_size
        == params_dict["num_attention_heads"] * self_attention.attention_head_size
    )

    assert self_attention.query.in_features == params_dict["hidden_size"]
    assert self_attention.key.in_features == params_dict["hidden_size"]
    assert self_attention.value.in_features == params_dict["hidden_size"]

    assert self_attention.dropout == params_dict["dropout"]


@pytest.mark.parametrize(
    "pretrained_name, relevant_module",
    [
        ("bert-base-cased", "bert.encoder.layer.0.attention.self"),
        ("google/electra-base-generator", "electra.encoder.layer.0.attention.self"),
        ("distilbert-base-uncased", "distilbert.transformer.layer.0.attention"),
    ],
)
def test_loading_from_pretrained_weights_using_model_name(pretrained_name, relevant_module):
    torch.manual_seed(1234)
    module = SelfAttention.from_pretrained_module(pretrained_name, relevant_module=relevant_module)

    torch.manual_seed(1234)
    pretrained_module = dict(AutoModel.from_pretrained(pretrained_name).named_modules())[
        # Module name will exclude the top-level part (e.g. 'bert.', 'electra.') for some reason.
        relevant_module[relevant_module.index(".") + 1 :]
    ]

    batch_size = 2
    seq_len = 3
    dim = module.query.in_features
    hidden_states = torch.randn(batch_size, seq_len, dim)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 1]])[:, None, None, :]

    # setting to eval mode to avoid non-deterministic dropout.
    module = module.eval()
    pretrained_module = pretrained_module.eval()

    torch.manual_seed(1234)
    output = module(hidden_states, attention_mask=attention_mask.squeeze()).hidden_states
    if "distilbert" in pretrained_name:
        torch.manual_seed(1234)
        hf_output = pretrained_module(
            hidden_states, hidden_states, hidden_states, mask=attention_mask
        )[0]
    else:
        # The attn_mask is processed outside the self attention module in HF bert models.
        attention_mask = (~(attention_mask == 1)) * min_value_of_dtype(hidden_states.dtype)
        torch.manual_seed(1234)
        hf_output = pretrained_module(hidden_states, attention_mask=attention_mask)[0]

    assert torch.allclose(output, hf_output)
