import math

import torch
from torch.nn import Parameter
from overrides import overrides
from allennlp.modules.attention.legacy_attention import Attention
from allennlp.nn import util
from allennlp.nn.activations import Activation
from allennlp.common.params import Params


@Attention.register("linear")
class LinearAttention(Attention):
    """
    This ``Attention`` module performs a dot product between a vector of weights and some
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
                 activation: Activation = Activation.by_name('linear')(),
                 normalize: bool = True) -> None:
        super().__init__(normalize)
        self._combination = combination
        combined_dim = util.get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    @overrides
    def _forward_internal(self,
                          vector: torch.Tensor,
                          matrix: torch.Tensor,
                          matrix_mask: torch.Tensor = None) -> torch.Tensor:
        # TODO(mattg): Remove the need for this tiling.
        # https://github.com/allenai/allennlp/pull/1235#issuecomment-391540133
        tiled_vector = vector.unsqueeze(1).expand(vector.size()[0],
                                                  matrix.size()[1],
                                                  vector.size()[1])

        combined_tensors = util.combine_tensors(self._combination, [tiled_vector, matrix])
        dot_product = torch.matmul(combined_tensors, self._weight_vector)
        return self._activation(dot_product + self._bias)

    @classmethod
    def from_params(cls, params: Params) -> 'Attention':
        tensor_1_dim = params.pop_int("tensor_1_dim")
        tensor_2_dim = params.pop_int("tensor_2_dim")
        combination = params.pop("combination", "x,y")
        activation = Activation.by_name(params.pop("activation", "linear"))()
        normalize = params.pop_bool('normalize', True)
        params.assert_empty(cls.__name__)
        return cls(normalize=normalize,
                   tensor_1_dim=tensor_1_dim,
                   tensor_2_dim=tensor_2_dim,
                   combination=combination,
                   activation=activation)
