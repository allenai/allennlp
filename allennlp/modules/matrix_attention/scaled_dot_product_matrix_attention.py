import math

import torch
from overrides import overrides

from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("scaled_dot_product")
class ScaledDotProductMatrixAttention(DotProductMatrixAttention):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using a dot
    product. Scales the result by the size of the embeddings.

    Registered as a `MatrixAttention` with name "scaled_dot_product".
    """

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        return super().forward(matrix_1, matrix_2) / math.sqrt(matrix_1.size(-1))
