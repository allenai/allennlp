from typing import Optional
import math
import torch

from allennlp.common import FromParams


class Attention(torch.nn.Module, FromParams):
    # Base class for different attention scoring mechanisms.
    def __init__(self, hidden_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, *args):
        return NotImplementedError


class GeneralAttention(Attention):
    # Reference [Effective Approaches to Attention-based Neural Machine Translation (Luong et al, 2015)]
    # (https://api.semanticscholar.org/CorpusID:1998416)
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size)
        self.Wa = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:  # type: ignore
        scores = self.Wa(query)
        scores = torch.matmul(scores, key.transpose(-1, -2))
        return scores


class AdditiveAttention(Attention):
    # Also known as: ConcatAttention / Bahdanau Attention.
    # Reference: [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al, 2015)]
    # (https://api.semanticscholar.org/CorpusID:11212020)
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size)
        self.Wa = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.va = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, query: torch.tensor, key: torch.tensor) -> torch.Tensor:  # type: ignore
        concatenated = torch.cat([query, key], dim=1)
        scores = self.va(torch.tanh(self.Wa(concatenated)))
        return scores


class DotProduct(Attention):
    # Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    # (https://api.semanticscholar.org/CorpusID:13756489)
    # Note: This does not have any parameters; it is a module for uniformity's sake.
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:  # type: ignore
        scores = torch.matmul(query, key.transpose(-1, -2))
        return scores


class ScaledDotProduct(Attention):
    # Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    # (https://api.semanticscholar.org/CorpusID:13756489)
    # Note: This does not have any parameters; it is a module for uniformity's sake.
    def __init__(self):
        super().__init__()

    def forward(  # type: ignore
        self, query: torch.Tensor, key: torch.Tensor, attention_head_size: int
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(attention_head_size)
        return scores


class ContentBaseAttention(Attention):
    # Reference: [Neural Turing Machines (Graves et al, 2014)]
    # (https://api.semanticscholar.org/CorpusID:15299054)
    # Note: This does not have any parameters; it is a module for uniformity's sake.
    def __init__(self):
        super().__init__()
        self.cos = torch.nn.CosineSimilarity(dim=1)  # TODO: check dim.

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:  # type: ignore
        scores = self.cos(query, key)
        return scores


def attention_map():
    attn_map = {}
    attn_map["general"] = GeneralAttention
    attn_map["additive"] = AdditiveAttention
    attn_map["dot_product"] = DotProduct
    attn_map["scaled_dot_product"] = ScaledDotProduct
    attn_map["content_base"] = ContentBaseAttention
    return attn_map


ATTN_MAP = attention_map()
