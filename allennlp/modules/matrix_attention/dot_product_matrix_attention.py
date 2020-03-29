import torch
from overrides import overrides

from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("dot_product")
class DotProductMatrixAttention(MatrixAttention):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using a dot
    product.

    Registered as a `MatrixAttention` with name "dot_product".
    """

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        return matrix_1.bmm(matrix_2.transpose(2, 1))
