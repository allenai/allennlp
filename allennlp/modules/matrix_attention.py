"""
A ``Module`` that takes two matrices as input and returns a matrix of attentions.
"""

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.modules.similarity_functions import DotProductSimilarity, SimilarityFunction


class MatrixAttention(torch.nn.Module):
    '''
    This ``Module`` takes two matrices as input and returns a matrix of attentions.

    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores.  Because these scores are unnormalized, we don't take a mask as input; it's up to the
    caller to deal with masking properly when this output is used.

    By default similarity is computed with a dot product, but you can alternatively use a
    parameterized similarity function if you wish.

    This is largely similar to using ``TimeDistributed(Attention)``, except the result is
    unnormalized.  You should use this instead of ``TimeDistributed(Attention)`` if you want to
    compute multiple normalizations of the attention matrix.

    Input:
        - matrix_1: ``(batch_size, num_rows_1, embedding_dim)``
        - matrix_2: ``(batch_size, num_rows_2, embedding_dim)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``

    Parameters
    ----------
    similarity_function: ``SimilarityFunction``, optional (default=``DotProductSimilarity``)
        The similarity function to use when computing the attention.
    '''
    def __init__(self, similarity_function: SimilarityFunction = None) -> None:
        super(MatrixAttention, self).__init__()

        self._similarity_function = similarity_function or DotProductSimilarity()

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])
        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)

    @classmethod
    def from_params(cls, params: Params) -> 'MatrixAttention':
        similarity_function = SimilarityFunction.from_params(params.pop('similarity_function', {}))
        return cls(similarity_function=similarity_function)
