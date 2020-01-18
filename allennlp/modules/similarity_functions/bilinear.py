from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.nn import Activation


@SimilarityFunction.register("bilinear")
class BilinearSimilarity(SimilarityFunction):
    """
    This similarity function performs a bilinear transformation of the two input vectors.  This
    function has a matrix of weights `W` and a bias `b`, and the similarity between two vectors
    `x` and `y` is computed as `x^T W y + b`.

    # Parameters

    tensor_1_dim : `int`
        The dimension of the first tensor, `x`, described above.  This is `x.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build the weight matrix correctly.
    tensor_2_dim : `int`
        The dimension of the second tensor, `y`, described above.  This is `y.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build the weight matrix correctly.
    activation : `Activation`, optional (default=linear (i.e. no activation))
        An activation function applied after the `x^T W y + b` calculation.  Default is no
        activation.
    """

    def __init__(self, tensor_1_dim: int, tensor_2_dim: int, activation: Activation = None) -> None:
        super().__init__()
        self._weight_matrix = Parameter(torch.Tensor(tensor_1_dim, tensor_2_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation or Activation.by_name("linear")()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        intermediate = torch.matmul(tensor_1, self._weight_matrix)
        result = (intermediate * tensor_2).sum(dim=-1)
        return self._activation(result + self._bias)
