import torch
from overrides import overrides
from allennlp.modules.attention.legacy_attention import Attention


@Attention.register("dot_product")
class DotProductAttention(Attention):
    """
    Computes attention between a vector and a matrix using dot product.
    """
    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)
