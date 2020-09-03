from typing import Union
import torch

from allennlp.common import FromParams

from transformers.modeling_bert import ACT2FN


class BertIntermediate(torch.nn.Module, FromParams):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Union[str, torch.nn.Module],  # TODO: restrict to activation
    ):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, intermediate_size)
        if isinstance(activation, str):
            self.intermediate_act_fn = ACT2FN[activation]
        else:
            self.intermediate_act_fn = activation

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
