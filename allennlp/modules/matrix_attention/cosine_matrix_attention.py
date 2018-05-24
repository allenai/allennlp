"""
A ``Module`` that takes two matrices as input and returns a matrix of attentions.
"""

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("cosine")
class CosineMatrixAttention(MatrixAttention):

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        a_norm = matrix_1 / (matrix_1.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        b_norm = matrix_2 / (matrix_2.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        return torch.bmm(a_norm, b_norm.transpose(-1, -2))

    @classmethod
    def from_params(cls, params: Params):
        CosineMatrixAttention()
