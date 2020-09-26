import math

import torch
from torch.nn import Parameter
from overrides import overrides

from allennlp.nn import util
from allennlp.nn.activations import Activation
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("linear")
class LinearMatrixAttention(MatrixAttention):
    """
    This `MatrixAttention` takes two matrices as input and returns a matrix of attentions
    by performing a dot product between a vector of weights and some
    combination of the two input matrices, followed by an (optional) activation function.  The
    combination used is configurable.

    If the two vectors are `x` and `y`, we allow the following kinds of combinations : `x`,
    `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give `x,y,x*y` as the `combination` parameter to this class.  The computed similarity
    function would then be `w^T [x; y; x*y] + b`, where `w` is a vector of weights, `b` is a
    bias parameter, and `[;]` is vector concatenation.

    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.

    Registered as a `MatrixAttention` with name "linear".

    # Parameters

    tensor_1_dim : `int`, required
        The dimension of the first tensor, `x`, described above.  This is `x.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : `int`, required
        The dimension of the second tensor, `y`, described above.  This is `y.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : `str`, optional (default=`"x,y"`)
        Described above.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `w^T * [x;y] + b` calculation.  Default is
        linear, i.e. no activation.
    """

    def __init__(
        self,
        tensor_1_dim: int,
        tensor_2_dim: int,
        combination: str = "x,y",
        activation: Activation = None,
    ) -> None:
        super().__init__()
        self._combination = combination
        combined_dim = util.get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation or Activation.by_name("linear")()
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        combined_tensors = util.combine_tensors_and_multiply(
            self._combination, [matrix_1.unsqueeze(2), matrix_2.unsqueeze(1)], self._weight_vector
        )
        return self._activation(combined_tensors + self._bias)
