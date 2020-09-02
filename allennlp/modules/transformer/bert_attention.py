import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.self_attention import SelfAttention
from allennlp.modules.transformer.bert_self_output import BertSelfOutput


class BertAttention(torch.nn.Module, FromParams):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_dropout)
        self.output = BertSelfOutput(hidden_size, hidden_dropout)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output
