"""
A ``Module`` that takes two matrices as input and returns a matrix of attentions.
"""

import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("dot_product")
class DotProductMatrixAttention(MatrixAttention):

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return matrix_1.bmm(matrix_2.transpose(2, 1))

    @classmethod
    def from_params(cls, params: Params):
        return DotProductMatrixAttention()
