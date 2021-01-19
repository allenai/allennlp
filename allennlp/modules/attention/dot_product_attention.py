import torch
from overrides import overrides
from allennlp.modules.attention.attention import Attention


@Attention.register("dot_product")
class DotProductAttention(Attention):
    """
    Computes attention between a vector and a matrix using dot product.

    Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "dot_product".
    """

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)
