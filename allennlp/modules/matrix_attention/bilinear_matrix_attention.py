from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import Activation


@MatrixAttention.register("bilinear")
class BilinearMatrixAttention(MatrixAttention):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights ``W`` and a bias ``b``, and the similarity between the two matrices ``X``
    and ``Y`` is computed as ``X W Y^T + b``.

    Parameters
    ----------
    matrix_1_dim : ``int``
        The dimension of the matrix ``X``, described above.  This is ``X.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : ``int``
        The dimension of the matrix ``Y``, described above.  This is ``Y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``X W Y^T + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 activation: Activation) -> None:
        super().__init__()
        self._weight_matrix = Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        intermediate = matrix_1.bmm(self._weight_matrix.unsqueeze(0))
        return self._activation(intermediate.bmm(matrix_2.transpose(1, 2)) + self._bias)

    @classmethod
    def from_params(cls, params: Params):
        matrix_1_dim = params.pop_int("matrix_1_dim")
        matrix_2_dim = params.pop_int("matrix_2_dim")
        activation = Activation.by_name(params.pop("activation", "linear"))()
        params.assert_empty(cls.__name__)
        return BilinearMatrixAttention(matrix_1_dim=matrix_1_dim,
                                       matrix_2_dim=matrix_2_dim,
                                       activation=activation)
