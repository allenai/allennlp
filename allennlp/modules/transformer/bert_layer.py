from typing import Union

import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.bert_attention import BertAttention
from allennlp.modules.transformer.bert_intermediate import BertIntermediate
from allennlp.modules.transformer.output_layer import OutputLayer


class BertLayer(torch.nn.Module, FromParams):
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
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
        )
        self.intermediate = BertIntermediate(
            hidden_size=hidden_size, intermediate_size=intermediate_size, activation=activation
        )
        self.output = OutputLayer(
            input_size=intermediate_size, hidden_size=hidden_size, dropout=hidden_dropout
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
