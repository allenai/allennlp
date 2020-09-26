from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation


@Attention.register("bilinear")
class BilinearAttention(Attention):
    """
    Computes attention between a vector and a matrix using a bilinear attention function.  This
    function has a matrix of weights `W` and a bias `b`, and the similarity between the vector
    `x` and the matrix `y` is computed as `x^T W y + b`.

    Registered as an `Attention` with name "bilinear".

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `x^T W y + b` calculation.  Default is
        linear, i.e. no activation.
    normalize : `bool`, optional (default=`True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(
        self,
        vector_dim: int,
        matrix_dim: int,
        activation: Activation = None,
        normalize: bool = True,
    ) -> None:
        super().__init__(normalize)
        self._weight_matrix = Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation or Activation.by_name("linear")()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        intermediate = vector.mm(self._weight_matrix).unsqueeze(1)
        return self._activation(intermediate.bmm(matrix.transpose(1, 2)).squeeze(1) + self._bias)
