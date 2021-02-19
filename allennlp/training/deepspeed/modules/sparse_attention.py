from typing import Optional, Union
from overrides import overrides
from copy import deepcopy

from allennlp.common import Registrable

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLayer

from deepspeed.ops.sparse_attention import (
    BertSparseSelfAttention,
    SparsityConfig,
    DenseSparsityConfig,
    FixedSparsityConfig,
    VariableSparsityConfig,
    BigBirdSparsityConfig,
    BSLongformerSparsityConfig,
)

import torch
import warnings


class SparseSelfAttentionLayer(BertSparseSelfAttention):
    @overrides
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor], *args, **kwargs
    ):
        extras = (*args, *kwargs.values())
        if not all(arg is None for arg in extras):
            warnings.warn("SparseSelfAttentionLayer only accepts hidden_states and attention_mask.")

        return (super().forward(hidden_states, attention_mask),)


def replace_self_attention(
    model: torch.nn.Module,
    sparsity_config: SparsityConfig,
    model_config: Union[BertConfig, RobertaConfig] = None,
):
    # Largely follows these:
    # https://github.com/microsoft/DeepSpeed/blob/c5b3f40e8481748f9658a19c2df1f17c5b579919/deepspeed/module_inject/inject.py#L6
    # https://github.com/microsoft/DeepSpeed/blob/c5b3f40e8481748f9658a19c2df1f17c5b579919/deepspeed/ops/sparse_attention/sparse_attention_utils.py#L85

    config = model_config or model.config
    assert isinstance(
        config, (BertConfig, RobertaConfig)
    ), "Only BERT and RoBERTa are currently supported by Deepspeed."

    for name, layer in model.named_children():
        if isinstance(layer, (BertLayer, RobertaLayer)):
            deepspeed_sparse_self_attn = SparseSelfAttentionLayer(config, sparsity_config)
            deepspeed_sparse_self_attn.query = layer.attention.self.query
            deepspeed_sparse_self_attn.key = layer.attention.self.key
            deepspeed_sparse_self_attn.value = layer.attention.self.value

            layer.attention.self = deepspeed_sparse_self_attn
            setattr(model, name, deepcopy(layer))
        else:
            replace_self_attention(layer, sparsity_config, model_config=config)

    return model


class _SparsityConfig(Registrable, SparsityConfig):
    default_implementation = "base"


_SparsityConfig.register("base")(SparsityConfig)
_SparsityConfig.register("dense")(DenseSparsityConfig)
_SparsityConfig.register("fixed")(FixedSparsityConfig)
_SparsityConfig.register("variable")(VariableSparsityConfig)
_SparsityConfig.register("bigbird")(BigBirdSparsityConfig)
_SparsityConfig.register("longformer")(BSLongformerSparsityConfig)
