from typing import Union, Optional, Dict

import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.transformer_module import TransformerModule

from allennlp.modules.transformer.activation_layer import ActivationLayer
from allennlp.modules.transformer.self_attention import SelfAttention
from allennlp.modules.transformer.output_layer import OutputLayer


class AttentionLayer(TransformerModule, FromParams):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_dropout)
        self.output = OutputLayer(hidden_size, hidden_size, hidden_dropout)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class TransformerLayer(TransformerModule, FromParams):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        activation: Union[str, torch.nn.Module],  # TODO: restrict to activation?
    ):
        super().__init__()
        self.attention = AttentionLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
        )
        self.intermediate = ActivationLayer(
            hidden_size=hidden_size, intermediate_size=intermediate_size, activation=activation
        )
        self.output = OutputLayer(
            input_size=intermediate_size, hidden_size=hidden_size, dropout=hidden_dropout
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    @classmethod
    def from_pretrained_module(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):

        submodules = cls._get_mapped_submodules(pretrained_module)

        hidden_size = submodules["attention.self.query"].in_features
        num_attention_heads = submodules["attention.self"].num_attention_heads
        attention_dropout = submodules["attention.self.dropout"].p
        hidden_dropout = submodules["attention.output.dropout"].p
        intermediate_size = submodules["intermediate.dense"].out_features
        activation = submodules["intermediate"].intermediate_act_fn

        module = cls(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
            activation,
        )

        module._load_from_pretrained_module(pretrained_module)

        return module
