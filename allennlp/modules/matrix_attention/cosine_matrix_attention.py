"""
A ``Module`` that takes two matrices as input and returns a matrix of attentions.
"""

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("cosine")
class CosineMatrixAttention(MatrixAttention):

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return F.cosine_similarity(matrix_1, matrix_1)

    @classmethod
    def from_params(cls, params: Params):
        CosineMatrixAttention()
