import torch
from overrides import overrides
from allennlp.modules.attention.attention import Attention
import math


@Attention.register("scaled_dot_product")
class ScaledDotProductAttention(Attention):
    """
    Computes attention between a vector and a matrix using scaled dot product.

    This attention is often referred to as "Scaled Dot-Product Attention". It was introduced in
    <https://arxiv.org/abs/1706.03762> by Vaswani et al.

    Registered as an `Attention` with name "scaled_dot_product".
    """

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1) / math.sqrt(matrix.size(-1))
