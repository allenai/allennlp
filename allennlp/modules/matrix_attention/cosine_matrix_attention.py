import torch
from overrides import overrides

from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util


@MatrixAttention.register("cosine")
class CosineMatrixAttention(MatrixAttention):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using cosine
    similarity.

    Registered as a `MatrixAttention` with name "cosine".
    """

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        a_norm = matrix_1 / (
            matrix_1.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix_1.dtype)
        )
        b_norm = matrix_2 / (
            matrix_2.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix_2.dtype)
        )
        return torch.bmm(a_norm, b_norm.transpose(-1, -2))
