import torch
from allennlp.common.params import Params

from allennlp.common.registrable import Registrable


class MatrixAttention(torch.nn.Module, Registrable):
    """
    This ``Attention`` takes two matrices as input and returns a matrix of attentions.

    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores.  Because these scores are unnormalized, we don't take a mask as input; it's up to the
    caller to deal with masking properly when this output is used.

    Input:
        - matrix_1: ``(batch_size, num_rows_1, embedding_dim)``
        - matrix_2: ``(batch_size, num_rows_2, embedding_dim)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``
    """
    def forward(self,  # pylint: disable=arguments-differ
                matrix_1: torch.Tensor,
                matrix_2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'MatrixAttention':
        clazz = cls.by_name(params.pop_choice("type", cls.list_available()))
        return clazz.from_params(params)
