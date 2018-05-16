"""
A ``Module`` that takes two matrices as input and returns a matrix of attentions.
"""

import torch
from allennlp.common.params import Params

from allennlp.common.registrable import Registrable

class MatrixAttention(torch.nn.Module, Registrable):
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
    '''
    @classmethod
    def from_params(cls, params: Params) -> 'MatrixAttention':
        clazz = cls.by_name(params.pop_choice("type", cls.list_available()))
        return clazz.from_params(params)