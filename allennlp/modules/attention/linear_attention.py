import math

import torch
from torch.nn import Parameter

from allennlp.nn.util import masked_softmax
from overrides import overrides
from allennlp.modules.attention.attention import Attention
from allennlp.nn import util
from allennlp.nn.activations import Activation
from allennlp.common.params import Params

@Attention.register("linear")
class LinearAttention(Attention):

    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 combination: str = 'x,y',
                 activation: Activation = Activation.by_name('linear')(),
                 normalize: bool = True) -> None:
        super(LinearAttention, self).__init__()
        self._normalize = normalize
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
    def forward(self,  # pylint: disable=arguments-differ
                vector: torch.Tensor,
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None) -> torch.Tensor:
        tiled_vector = vector.unsqueeze(1).expand(vector.size()[0],
                                                  matrix.size()[1],
                                                  vector.size()[1])

        combined_tensors = util.combine_tensors(self._combination, [tiled_vector, matrix])
        dot_product = torch.matmul(combined_tensors, self._weight_vector)
        similarities = self._activation(dot_product + self._bias)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    @classmethod
    def from_params(cls, params: Params) -> 'Attention':
        normalize = params.pop_bool('normalize', True)
        params.assert_empty(cls.__name__)
        return cls(normalize=normalize)

