"""
A ``Module`` that takes two matrices as input and returns a matrix of attentions.
"""
import math

import torch

from allennlp.common import Params
from allennlp.modules.matrix_attention import MatrixAttention
from overrides import overrides

@MatrixAttention.register("dot_product_matrix_attention")
class DotProductMatrixAttention(MatrixAttention):

    def __init__(self) -> None:
        super(DotProductMatrixAttention, self).__init__()

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return matrix_1.bmm(matrix_2.transpose(2,1))

    @classmethod
    def from_params(cls, params: Params):
        return DotProductMatrixAttention()

