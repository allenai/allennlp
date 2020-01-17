import math

from overrides import overrides
import torch

from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction


@SimilarityFunction.register("dot_product")
class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.

    # Parameters

    scale_output : `bool`, optional
        If `True`, we will scale the output by `math.sqrt(tensor.size(-1))`, to reduce the
        variance in the result.
    """

    def __init__(self, scale_output: bool = False) -> None:
        super().__init__()
        self._scale_output = scale_output

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result
