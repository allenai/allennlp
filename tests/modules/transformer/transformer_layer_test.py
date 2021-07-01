import copy

import torch
import pytest
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertLayer
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaLayer
from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraAttention, ElectraLayer

from allennlp.common import Params, cached_transformers
from allennlp.common.testing import run_distributed_test
from allennlp.modules.transformer import (
    AttentionLayer,
    TransformerLayer,
)


def teardown_module(function):
    cached_transformers._clear_caches()


ATTENTION_PARAMS_DICT = {
    "hidden_size": 6,
    "num_attention_heads": 2,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.2,
}


@pytest.fixture
def attention_params():
    return Params(copy.deepcopy(ATTENTION_PARAMS_DICT))


def test_attention(attention_params):
    attention_layer = AttentionLayer.from_params(attention_params.duplicate()).eval()

    assert attention_layer.self.num_attention_heads == attention_params["num_attention_heads"]
    assert attention_layer.self.attention_head_size == int(
        attention_params["hidden_size"] / attention_params["num_attention_heads"]
    )
    assert (
        attention_layer.self.all_head_size
        == attention_params["num_attention_heads"] * attention_layer.self.attention_head_size
    )
    assert attention_layer.self.query.in_features == attention_params["hidden_size"]
    assert attention_layer.self.key.in_features == attention_params["hidden_size"]
    assert attention_layer.self.value.in_features == attention_params["hidden_size"]
    assert attention_layer.self.dropout == attention_params["attention_dropout"]

    assert attention_layer.output.dense.in_features == attention_params["hidden_size"]
    assert attention_layer.output.dense.out_features == attention_params["hidden_size"]
    assert attention_layer.output.layer_norm.normalized_shape[0] == attention_params["hidden_size"]
    assert attention_layer.output.dropout.p == attention_params["hidden_dropout"]

    attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
    attention_layer(torch.randn(2, 3, 6), attention_mask=attention_mask)


def get_attention_modules():
    params = copy.deepcopy(ATTENTION_PARAMS_DICT)
    params["attention_probs_dropout_prob"] = params.pop("attention_dropout")
    params["hidden_dropout_prob"] = params.pop("hidden_dropout")

    torch.manual_seed(1234)
    yield "bert", BertAttention(BertConfig(**params)).eval()

    torch.manual_seed(1234)
    yield "roberta", RobertaAttention(RobertaConfig(**params)).eval()

    torch.manual_seed(1234)
    yield "electra", ElectraAttention(ElectraConfig(**params)).eval()


@pytest.mark.parametrize("module_name, hf_module", get_attention_modules())
def test_attention_matches_huggingface(attention_params, module_name, hf_module):
    hidden_states = torch.randn(2, 3, 6)
    attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])

    attention = AttentionLayer.from_params(attention_params).eval()
    state_dict = attention._get_mapped_state_dict(hf_module.state_dict())
    attention.load_state_dict(state_dict)

    torch.manual_seed(1234)
    output = attention(hidden_states, attention_mask=attention_mask)
    # We do this because bert, roberta, electra process the attention_mask at the model level.
    attention_mask_hf = (attention_mask == 0).view((2, 1, 1, 3)).expand(2, 2, 3, 3) * -10e5

    torch.manual_seed(1234)
    hf_output = hf_module(hidden_states, attention_mask=attention_mask_hf)

    assert torch.allclose(output.hidden_states, hf_output[0])


@pytest.mark.parametrize(
    "pretrained_name, relevant_top_level_module",
    [
        ("bert-base-cased", "bert"),
        ("epwalsh/bert-xsmall-dummy", None),
    ],
)
def test_attention_from_pretrained(pretrained_name, relevant_top_level_module):
    torch.manual_seed(1234)
    pretrained = cached_transformers.get(pretrained_name, False).eval()

    if "distilbert" in pretrained_name:
        encoder = pretrained.transformer
    else:
        encoder = pretrained.encoder
    # Hacky way to get a bert layer.
    pretrained_module = list(encoder.layer.modules())[1].attention

    torch.manual_seed(1234)
    module = AttentionLayer.from_pretrained_module(
        pretrained_name,
        relevant_module=None
        if relevant_top_level_module is None
        else f"{relevant_top_level_module}.encoder.layer.0.attention",
    ).eval()

    batch_size = 2
    seq_length = 15
    hidden_size = module.self.query.in_features

    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    attention_mask = torch.randint(0, 2, (batch_size, seq_length))
    attention_mask_hf = attention_mask[:, None, None, :]
    attention_mask_hf = (1.0 - attention_mask_hf) * -10e5

    torch.manual_seed(1234)
    output = module(hidden_states, attention_mask=attention_mask.squeeze()).hidden_states

    torch.manual_seed(1234)
    hf_output = pretrained_module(hidden_states, attention_mask=attention_mask_hf)[0]

    assert torch.allclose(output, hf_output, atol=1e-04)


LAYER_PARAMS_DICT = {
    "hidden_size": 6,
    "intermediate_size": 3,
    "num_attention_heads": 2,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.2,
    "activation": "relu",
}


@pytest.fixture
def layer_params():
    return Params(copy.deepcopy(LAYER_PARAMS_DICT))


def test_layer(layer_params):
    transformer_layer = TransformerLayer.from_params(layer_params.duplicate()).eval()

    assert (
        transformer_layer.attention.self.num_attention_heads == layer_params["num_attention_heads"]
    )
    assert transformer_layer.attention.self.attention_head_size == int(
        layer_params["hidden_size"] / layer_params["num_attention_heads"]
    )
    assert (
        transformer_layer.attention.self.all_head_size
        == layer_params["num_attention_heads"]
        * transformer_layer.attention.self.attention_head_size
    )
    assert transformer_layer.attention.self.query.in_features == layer_params["hidden_size"]
    assert transformer_layer.attention.self.key.in_features == layer_params["hidden_size"]
    assert transformer_layer.attention.self.value.in_features == layer_params["hidden_size"]
    assert transformer_layer.attention.self.dropout == layer_params["attention_dropout"]

    assert transformer_layer.attention.output.dense.in_features == layer_params["hidden_size"]
    assert transformer_layer.attention.output.dense.out_features == layer_params["hidden_size"]
    assert (
        transformer_layer.attention.output.layer_norm.normalized_shape[0]
        == layer_params["hidden_size"]
    )
    assert transformer_layer.attention.output.dropout.p == layer_params["hidden_dropout"]

    assert transformer_layer.intermediate.dense.in_features == layer_params["hidden_size"]
    assert transformer_layer.intermediate.dense.out_features == layer_params["intermediate_size"]

    assert transformer_layer.output.dense.in_features == layer_params["intermediate_size"]
    assert transformer_layer.output.dense.out_features == layer_params["hidden_size"]

    assert transformer_layer.output.layer_norm.normalized_shape[0] == layer_params["hidden_size"]

    assert transformer_layer.output.dropout.p == layer_params["hidden_dropout"]

    attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
    transformer_layer(torch.randn(2, 3, 6), attention_mask=attention_mask)

    with pytest.raises(AssertionError):
        transformer_layer(
            torch.randn(2, 3, 6),
            attention_mask=attention_mask,
            encoder_hidden_states=torch.randn(2, 3, 6),
        )


def test_layer_with_cross_attention(layer_params):
    layer_params["add_cross_attention"] = True

    transformer_layer = TransformerLayer.from_params(layer_params).eval()
    assert hasattr(transformer_layer, "cross_attention")

    attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
    transformer_layer(
        torch.randn(2, 3, 6),
        attention_mask=attention_mask,
        encoder_hidden_states=torch.randn(2, 3, 6),
    )


def get_layer_modules():
    params = copy.deepcopy(LAYER_PARAMS_DICT)
    params["attention_probs_dropout_prob"] = params.pop("attention_dropout")
    params["hidden_dropout_prob"] = params.pop("hidden_dropout")
    params["hidden_act"] = params.pop("activation")

    torch.manual_seed(1234)
    yield "bert", BertLayer(BertConfig(**params)).eval()

    torch.manual_seed(1234)
    yield "roberta", RobertaLayer(RobertaConfig(**params)).eval()

    torch.manual_seed(1234)
    yield "electra", ElectraLayer(ElectraConfig(**params)).eval()


@pytest.mark.parametrize("module_name, hf_module", get_layer_modules())
def test_layer_matches_huggingface(layer_params, module_name, hf_module):
    layer = TransformerLayer.from_params(layer_params).eval()
    state_dict = layer._get_mapped_state_dict(hf_module.state_dict())
    layer.load_state_dict(state_dict)

    hidden_states = torch.randn(2, 3, 6)
    attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])

    torch.manual_seed(1234)
    output = layer(hidden_states, attention_mask=attention_mask)
    # We do this because bert, roberta, electra process the attention_mask at the model level.
    attention_mask_hf = (attention_mask == 0).view((2, 1, 1, 3)).expand(2, 2, 3, 3) * -10e5
    torch.manual_seed(1234)
    hf_output = hf_module(hidden_states, attention_mask=attention_mask_hf)

    assert torch.allclose(output.hidden_states, hf_output[0])


@pytest.mark.parametrize(
    "pretrained_name, relevant_top_level_module",
    [
        ("bert-base-cased", "bert"),
        ("epwalsh/bert-xsmall-dummy", None),
    ],
)
def test_layer_from_pretrained(pretrained_name, relevant_top_level_module):
    torch.manual_seed(1234)
    pretrained = cached_transformers.get(pretrained_name, False).eval()

    if "distilbert" in pretrained_name:
        encoder = pretrained.transformer
    else:
        encoder = pretrained.encoder
    # Hacky way to get a bert layer.
    pretrained_module = list(encoder.layer.modules())[1]

    torch.manual_seed(1234)
    module = TransformerLayer.from_pretrained_module(
        pretrained_name,
        relevant_module=None
        if relevant_top_level_module is None
        else f"{relevant_top_level_module}.encoder.layer.0",
    ).eval()

    batch_size = 2
    seq_length = 15
    hidden_size = module.attention.self.query.in_features

    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    attention_mask = torch.randint(0, 2, (batch_size, seq_length))
    attention_mask_hf = attention_mask[:, None, None, :]
    attention_mask_hf = (1.0 - attention_mask_hf) * -10e5

    torch.manual_seed(1234)
    output = module(hidden_states, attention_mask=attention_mask.squeeze()).hidden_states

    torch.manual_seed(1234)
    hf_output = pretrained_module(hidden_states, attention_mask=attention_mask_hf)[0]

    assert torch.allclose(output, hf_output, atol=1e-04)


def _load_pretrained(global_rank, world_size, gpu_id):
    TransformerLayer.from_pretrained_module(
        "epwalsh/bert-xsmall-dummy",
    )


@pytest.mark.parametrize("test_func", [_load_pretrained])
def test_distributed(test_func):
    run_distributed_test([-1, -1], func=test_func, start_method="spawn")
