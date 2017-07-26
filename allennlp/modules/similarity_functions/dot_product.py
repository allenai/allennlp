from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.experiments import Registry
from allennlp.modules import SimilarityFunction


@Registry.register_similarity_function("dot_product")
class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors.  It has
    no parameters.
    """
    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        return (tensor_1 * tensor_2).sum(dim=-1).squeeze(dim=-1)

    @classmethod
    def from_params(cls, params: Params):
        params.assert_empty(cls.__name__)
        return cls()
