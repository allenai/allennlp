import copy
import torch
import pytest

from transformers import AutoModel

from allennlp.common import Params

from allennlp.modules.transformer.attention_module import T5Attention

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Attention as HFT5Attention
from allennlp.nn.util import min_value_of_dtype

PARAMS_DICT = {
    "hidden_size": 6,
    "num_heads": 2,
    "key_value_proj_dim": 3,
    "dropout": 0.0,
    "relative_attention_num_buckets": 2,
}


@pytest.fixture
def params_dict():
    return copy.deepcopy(PARAMS_DICT)


@pytest.fixture
def params(params_dict):
    return Params(params_dict)


@pytest.fixture
def t5_attention(params):
    return T5Attention.from_params(params.duplicate())


def test_can_construct_from_params(t5_attention, params_dict):

    assert t5_attention.num_attention_heads == params_dict["num_heads"]
    assert t5_attention.attention_head_size == params_dict["key_value_proj_dim"]

    assert (
        t5_attention.all_head_size == params_dict["num_heads"] * params_dict["key_value_proj_dim"]
    )

    assert t5_attention.query.in_features == params_dict["hidden_size"]
    assert t5_attention.key.in_features == params_dict["hidden_size"]
    assert t5_attention.value.in_features == params_dict["hidden_size"]
    assert t5_attention.output.in_features == params_dict["hidden_size"]

    assert t5_attention.dropout == params_dict["dropout"]


def test_forward_against_huggingface_output(params_dict):
    hidden_states = torch.randn(2, 3, 6)
    attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])

    hf_kwargs = {
        "d_model": params_dict["hidden_size"],
        "d_kv": params_dict["key_value_proj_dim"],
        "num_heads": params_dict["num_heads"],
        "relative_attention_num_buckets": params_dict["relative_attention_num_buckets"],
        "dropout_rate": params_dict["dropout"],
    }

    torch.manual_seed(1234)
    hf_module = HFT5Attention(T5Config(**hf_kwargs), has_relative_attention_bias=False)

    torch.manual_seed(1234)

    params = copy.deepcopy(params_dict)
    params["normalize"] = False  # only for this test, as HF does not normalize.
    t5_attention = T5Attention(**params)

    # setting to eval mode to avoid non-deterministic dropout.
    t5_attention = t5_attention.eval()
    hf_module = hf_module.eval()

    output = t5_attention.forward(hidden_states, mask=attention_mask)
    attention_mask_hf = (attention_mask == 0).view((2, 1, 1, 3)).expand(
        2, 2, 3, 3
    ) * min_value_of_dtype(hidden_states.dtype)
    hf_output = hf_module.forward(hidden_states, mask=attention_mask_hf)

    hs = output.hidden_states

    assert torch.allclose(hs, hf_output[0])


@pytest.mark.parametrize(
    "pretrained_name, relevant_module",
    [
        ("t5-small", "encoder.block.0.layer.0.SelfAttention"),
    ],
)
def test_loading_from_pretrained_weights_using_model_name(pretrained_name, relevant_module):

    torch.manual_seed(1234)
    module = T5Attention.from_pretrained_module(pretrained_name, relevant_module=relevant_module)

    torch.manual_seed(1234)
    pretrained_module = dict(AutoModel.from_pretrained(pretrained_name).named_modules())[
        relevant_module
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
    output = module(hidden_states, mask=attention_mask.squeeze()).hidden_states

    # The attn_mask is processed outside the self attention module in HF bert models.
    attention_mask = (~(attention_mask == 1)) * min_value_of_dtype(hidden_states.dtype)
    torch.manual_seed(1234)
    hf_output = pretrained_module(hidden_states, mask=attention_mask)[0]

    assert torch.allclose(output, hf_output)
