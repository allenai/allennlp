import math

from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.similarity_function import SimilarityFunction
from allennlp.nn import Activation


@SimilarityFunction.register("linear")
class LinearSimilarity(SimilarityFunction):
    """
    This similarity function performs a dot product between a vector of weights and some
    combination of the two input vectors, followed by an (optional) activation function.  The
    combination used is configurable.

    If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations: ``x``,
    ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give ``x,y,x*y`` as the ``combination`` parameter to this class.  The computed similarity
    function would then be ``w^T [x; y; x*y] + b``, where ``w`` is a vector of weights, ``b`` is a
    bias parameter, and ``[;]`` is vector concatenation.

    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.

    Parameters
    ----------
    tensor_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : ``str``, optional (default="x,y")
        Described above.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 combination: str = 'x,y',
                 activation: Activation = Activation.by_name('linear')()) -> None:
        super(LinearSimilarity, self).__init__()
        self._combinations = combination.split(',')
        combined_dim = self._get_combined_dim(tensor_1_dim, tensor_2_dim)
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        combined_tensors = self._combine_tensors(tensor_1, tensor_2)
        dot_product = torch.matmul(combined_tensors, self._weight_vector)
        return self._activation(dot_product + self._bias)

    def _combine_tensors(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        combined_tensor = self._get_combination(self._combinations[0], tensor_1, tensor_2)
        for combination in self._combinations[1:]:
            to_concatenate = self._get_combination(combination, tensor_1, tensor_2)
            combined_tensor = torch.cat([combined_tensor, to_concatenate], dim=-1)
        return combined_tensor

    def _get_combination(self, combination: str, tensor_1, tensor_2):
        if combination == 'x':
            return tensor_1
        elif combination == 'y':
            return tensor_2
        else:
            if len(combination) != 3:
                raise ConfigurationError("Invalid combination: " + combination)
            first_tensor = self._get_combination(combination[0], tensor_1, tensor_2)
            second_tensor = self._get_combination(combination[2], tensor_1, tensor_2)
            operation = combination[1]
            if operation == '*':
                return first_tensor * second_tensor
            elif operation == '/':
                return first_tensor / second_tensor
            elif operation == '+':
                return first_tensor + second_tensor
            elif operation == '-':
                return first_tensor - second_tensor
            else:
                raise ConfigurationError("Invalid operation: " + operation)

    def _get_combined_dim(self, tensor_1_dim: int, tensor_2_dim: int) -> int:
        combination_dims = [self._get_combination_dim(combination, tensor_1_dim, tensor_2_dim)
                            for combination in self._combinations]
        return sum(combination_dims)

    def _get_combination_dim(self, combination: str, tensor_1_dim: int, tensor_2_dim: int) -> int:
        if combination == 'x':
            return tensor_1_dim
        elif combination == 'y':
            return tensor_2_dim
        else:
            if len(combination) != 3:
                raise ConfigurationError("Invalid combination: " + combination)
            first_tensor_dim = self._get_combination_dim(combination[0], tensor_1_dim, tensor_2_dim)
            second_tensor_dim = self._get_combination_dim(combination[2], tensor_1_dim, tensor_2_dim)
            operation = combination[1]
            if first_tensor_dim != second_tensor_dim:
                raise ConfigurationError("Tensor dims must match for operation \"{}\"".format(operation))
            return first_tensor_dim

    @classmethod
    def from_params(cls, params: Params) -> 'LinearSimilarity':
        tensor_1_dim = params.pop("tensor_1_dim")
        tensor_2_dim = params.pop("tensor_2_dim")
        combination = params.pop("combination", "x,y")
        activation = Activation.by_name(params.pop("activation", "linear"))()
        params.assert_empty(cls.__name__)
        return cls(tensor_1_dim=tensor_1_dim,
                   tensor_2_dim=tensor_2_dim,
                   combination=combination,
                   activation=activation)
