"""
An *attention* module that computes the similarity between
an input vector and the rows of a matrix.
"""

import torch

from overrides import overrides
from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_softmax


class Attention(torch.nn.Module, Registrable):
    """
    An `Attention` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
    rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
    and then (optionally) perform a softmax over rows using those computed similarities.


    Inputs:

    - vector: shape `(batch_size, embedding_dim)`
    - matrix: shape `(batch_size, num_rows, embedding_dim)`
    - matrix_mask: shape `(batch_size, num_rows)`, specifying which rows are just padding.

    Output:

    - attention: shape `(batch_size, num_rows)`.

    # Parameters

    normalize : `bool`, optional (default = `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self._normalize = normalize

    @overrides
    def forward(
        self, vector: torch.Tensor, matrix: torch.Tensor, matrix_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        similarities = self._forward_internal(vector, matrix)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
