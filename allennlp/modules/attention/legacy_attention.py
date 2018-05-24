"""
An *attention* module that computes the similarity between
an input vector and the rows of a matrix.
"""

import torch

from overrides import overrides
from allennlp.common import Params
from allennlp.modules.attention.attention import Attention
from allennlp.modules.similarity_functions import DotProductSimilarity, SimilarityFunction


@Attention.register("legacy")
class LegacyAttention(Attention):

    def __init__(self,
                 similarity_function: SimilarityFunction = None,
                 normalize: bool = True) -> None:
        super(LegacyAttention, self).__init__(normalize)
        self._similarity_function = similarity_function or DotProductSimilarity()

    @overrides
    def _forward_internal(self,
                          vector: torch.Tensor,
                          matrix: torch.Tensor,
                          matrix_mask: torch.Tensor = None) -> torch.Tensor:
        tiled_vector = vector.unsqueeze(1).expand(vector.size()[0],
                                                  matrix.size()[1],
                                                  vector.size()[1])
        return self._similarity_function(tiled_vector, matrix)

    @classmethod
    def from_params(cls, params: Params) -> 'Attention':
        similarity_function = SimilarityFunction.from_params(params.pop('similarity_function', {}))
        normalize = params.pop_bool('normalize', True)
        params.assert_empty(cls.__name__)
        return cls(similarity_function=similarity_function,
                   normalize=normalize)
