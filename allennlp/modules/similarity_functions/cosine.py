from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.experiments import Registry
from allennlp.modules import SimilarityFunction


@Registry.register_similarity_function("cosine")
class CosineSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the cosine similarity between each pair of vectors.  It has
    no parameters.
    """
    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        # TODO(mattg): remove the .expand_as() stuff once we upgrade to pytorch-0.2, which has
        # broadcasting implemented for this case.
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1).expand_as(tensor_1)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1).expand_as(tensor_2)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1).squeeze(dim=-1)

    @classmethod
    def from_params(cls, params: Params):
        params.assert_empty(cls.__name__)
        return cls()
