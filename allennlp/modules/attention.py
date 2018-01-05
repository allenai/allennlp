"""
An *attention* module that computes the similarity between
an input vector and the rows of a matrix.
"""

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.modules.similarity_functions import DotProductSimilarity, SimilarityFunction
from allennlp.nn.util import masked_softmax


class Attention(torch.nn.Module):
    """
    This ``Module`` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
    rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
    and then (optionally) perform a softmax over rows using those computed similarities.

    By default similarity is computed with a dot product, but you can alternatively use a
    parameterized similarity function if you wish.

    Inputs:

    - vector: shape ``(batch_size, embedding_dim)``
    - matrix: shape ``(batch_size, num_rows, embedding_dim)``
    - matrix_mask: shape ``(batch_size, num_rows)``, specifying which rows are just padding.

    Output:

    - attention: shape ``(batch_size, num_rows)``.

    Parameters
    ----------
    similarity_function : ``SimilarityFunction``, optional (default=``DotProductSimilarity``)
        The similarity function to use when computing the attention.
    normalize : ``bool``, optional (default: ``True``)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """
    def __init__(self,
                 similarity_function: SimilarityFunction = None,
                 normalize: bool = True) -> None:
        super(Attention, self).__init__()

        self._similarity_function = similarity_function or DotProductSimilarity()
        self._normalize = normalize

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                vector: torch.Tensor,
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None) -> torch.Tensor:
        tiled_vector = vector.unsqueeze(1).expand(vector.size()[0],
                                                  matrix.size()[1],
                                                  vector.size()[1])
        similarities = self._similarity_function(tiled_vector, matrix)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    @classmethod
    def from_params(cls, params: Params) -> 'Attention':
        similarity_function = SimilarityFunction.from_params(params.pop('similarity_function', {}))
        normalize = params.pop_bool('normalize', True)
        return cls(similarity_function=similarity_function,
                   normalize=normalize)
