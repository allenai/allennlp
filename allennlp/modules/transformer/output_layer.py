import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.transformer_module import TransformerModule


class OutputLayer(TransformerModule, FromParams):

    _huggingface_mapping = {"LayerNorm": "layer_norm"}

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.dense = torch.nn.Linear(input_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        dense_output = self.dense(hidden_states)
        dropout_output = self.dropout(dense_output)
        output = self.layer_norm(dropout_output + input_tensor)
        return output
