from typing import Union, Optional, Dict

import torch

from allennlp.common import FromParams

from allennlp.modules.util import replicate_layers
from allennlp.modules.transformer.transformer_layer import TransformerLayer
from allennlp.modules.transformer.transformer_module import TransformerModule


class TransformerBlock(TransformerModule, FromParams):
    """
    This module is the basic transformer block, which acts as an encoder.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)

    # Parameters

    num_hidden_layers : `int`
    hidden_size : `int`
    intermediate_size : `int`
    num_attention_heads : `int`
    attention_dropout : `float` (default = `0.0`)
        Dropout probability for the `SelfAttention` layer.
    hidden_dropout : `float` (default = `0.0`)
        Dropout probability for the `OutputLayer`.
    activation : `Union[str, torch.nn.Module]` (default = `"relu"`)
    """

    _huggingface_mapping = {"layer": "layers"}
    _relevant_module = "encoder"

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: Union[str, torch.nn.Module] = "relu",
    ):
        super().__init__()
        self._hidden_size = hidden_size
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """
        hidden_states : `torch.Tensor`
            Shape `batch_size x seq_len x hidden_dim`
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len`
        head_mask : `torch.BoolTensor`, optional
        output_attentions : `bool`
            Whether to also return the attention probabilities, default = `False`
        output_hidden_states : `bool`
            Whether to return the hidden_states for all layers, default = `False`
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # type: ignore

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

    @classmethod
    def _get_input_arguments(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)

        final_kwargs = {}

        final_kwargs["num_hidden_layers"] = len(submodules["layers"])

        final_kwargs["hidden_size"] = submodules["layers.0.attention.self.query"].in_features
        final_kwargs["num_attention_heads"] = submodules[
            "layers.0.attention.self"
        ].num_attention_heads
        final_kwargs["attention_dropout"] = submodules["layers.0.attention.self.dropout"].p
        final_kwargs["hidden_dropout"] = submodules["layers.0.attention.output.dropout"].p
        final_kwargs["intermediate_size"] = submodules["layers.0.intermediate.dense"].out_features
        final_kwargs["activation"] = submodules["layers.0.intermediate"].intermediate_act_fn

        final_kwargs.update(**kwargs)

        return final_kwargs

    @classmethod
    def from_pretrained_module(  # type: ignore
        cls,
        pretrained_module: Union[str, torch.nn.Module],
        num_hidden_layers: Optional[Union[int, range]] = None,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        final_kwargs = {}
        if num_hidden_layers is not None:
            if isinstance(num_hidden_layers, range):
                if mapping is None:
                    mapping = {}
                    for num_layer, mapped in enumerate(num_hidden_layers):
                        mapping[str(mapped)] = str(num_layer)
                final_kwargs["num_hidden_layers"] = len(num_hidden_layers)
            else:
                final_kwargs["num_hidden_layers"] = num_hidden_layers

        return super().from_pretrained_module(pretrained_module, source, mapping, **final_kwargs)
