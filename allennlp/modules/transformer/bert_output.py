import torch

from allennlp.common import FromParams


class BertOutput(torch.nn.Module, FromParams):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.dense = torch.nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states
