
import torch

from overrides import overrides
from allennlp.modules.attention.attention import Attention
from allennlp.modules.similarity_functions import DotProductSimilarity, SimilarityFunction


@Attention.register("legacy")
class LegacyAttention(Attention):
    """
    Computes attention between a vector and a matrix using a similarity function.
    This should be considered deprecated, as it consumes more memory than the specialized attention modules.
    """

    def __init__(self,
                 similarity_function: SimilarityFunction = None,
                 normalize: bool = True) -> None:
        super().__init__(normalize)
        self._similarity_function = similarity_function or DotProductSimilarity()

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        tiled_vector = vector.unsqueeze(1).expand(vector.size()[0],
                                                  matrix.size()[1],
                                                  vector.size()[1])
        return self._similarity_function(tiled_vector, matrix)
