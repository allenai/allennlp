"""
A ``Module`` that takes two matrices as input and returns a matrix of attentions.
"""

import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.modules.attention.attention import Attention


@Attention.register("dot_product")
class DotProductAttention(Attention):

    @overrides
    def forward(self,
                vector: torch.Tensor,
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return vector.bmm(matrix.transpose(2, 1)).squeeze(1)

    @classmethod
    def from_params(cls, params: Params):
        return DotProductAttention()
