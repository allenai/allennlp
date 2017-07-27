from typing import Callable

from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.experiments import Registry
from allennlp.modules import SimilarityFunction


@Registry.register_similarity_function("bilinear")
class BilinearSimilarity(SimilarityFunction):
    """
    This similarity function performs a bilinear transformation of the two input vectors.  This
    function has a matrix of weights ``W`` and a bias ``b``, and the similarity between two vectors
    ``x`` and ``y`` is computed as ``x^T W y + b``.

    Parameters
    ----------
    tensor_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build the weight matrix correctly.
    tensor_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build the weight matrix correctly.
    activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``lambda x: x``)
        An activation function applied after the ``x^T W y + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 activation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> None:
        super(BilinearSimilarity, self).__init__()
        self._weight_matrix = Parameter(torch.Tensor(tensor_1_dim, tensor_2_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        # The '@' operator here is torch.matmul, but that's only available in pytorch-0.2.
        # TODO(mattg): switch to using torch.matmul when a version of pytorch with it is released.
        # I think it's more clear and less magical, and won't require us to have to special case
        # the higher-order version.
        # Also, broadcasting this simple addition is only available in pytorch-0.2.  When that's
        # ready, change this back to `(dot_product + self._bias)`.
        if tensor_1.dim() <= 2:
            intermediate = tensor_1 @ self._weight_matrix
        else:
            view_args = [-1] + list(tensor_1.size()[-2:])
            reshaped_tensor = tensor_1.view(*view_args)
            unsqueezed_weight = self._weight_matrix.unsqueeze(0)
            reshaped_weight = unsqueezed_weight.expand(reshaped_tensor.size()[0],
                                                       self._weight_matrix.size()[0],
                                                       self._weight_matrix.size()[1])
            reshaped_intermediate = reshaped_tensor.bmm(reshaped_weight)
            view_args = tensor_1.size()[:-1] + self._weight_matrix.size()[1:]
            intermediate = reshaped_intermediate.view(*view_args)
        result = (intermediate * tensor_2).sum(dim=-1).squeeze(dim=-1)
        return self._activation(result + self._bias.expand_as(result))

    @classmethod
    def from_params(cls, params: Params):
        tensor_1_dim = params.pop("tensor_1_dim")
        tensor_2_dim = params.pop("tensor_2_dim")
        # TODO(mattg): figure out activation from_params.
        activation = lambda x: x
        params.assert_empty(cls.__name__)
        return cls(tensor_1_dim=tensor_1_dim,
                   tensor_2_dim=tensor_2_dim,
                   activation=activation)
