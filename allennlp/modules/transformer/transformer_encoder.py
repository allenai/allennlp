from typing import Union, Optional, Dict

import torch

from allennlp.common import FromParams

from allennlp.modules.util import replicate_layers
from allennlp.modules.transformer.transformer_layer import TransformerLayer
from allennlp.modules.transformer.transformer_module import TransformerModule


class TransformerEncoder(TransformerModule, FromParams):

    _huggingface_mapping = {"layers": "layer"}

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        activation: Union[str, torch.nn.Module],
    ):
        super().__init__()
        layer = TransformerLayer(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
            activation,
        )
        self.layers = replicate_layers(layer, num_hidden_layers)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        # FIX: forward doesn't work yet!
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

    @classmethod
    def _get_input_arguments(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):
        submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)

        kwargs = {}

        kwargs["num_hidden_layers"] = len(submodules["layers"])

        kwargs["hidden_size"] = submodules["layers.0.attention.self.query"].in_features
        kwargs["num_attention_heads"] = submodules["layers.0.attention.self"].num_attention_heads
        kwargs["attention_dropout"] = submodules["layers.0.attention.self.dropout"].p
        kwargs["hidden_dropout"] = submodules["layers.0.attention.output.dropout"].p
        kwargs["intermediate_size"] = submodules["layers.0.intermediate.dense"].out_features
        kwargs["activation"] = submodules["layers.0.intermediate"].intermediate_act_fn

        return kwargs
