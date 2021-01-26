import math
import torch
from overrides import overrides
from allennlp.modules.attention.attention import Attention


@Attention.register("scaled_dot_product")
class ScaledDotProductAttention(Attention):
    """
    Computes attention between two tensors using scaled dot product.
    # Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    # (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "scaled_dot_product".

    # Parameters

    scaling_factor : `int`, required
        The similarity score is scaled down by the `scaling_factor`.
    normalize : `bool`, optional (default=`True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, scaling_factor: int, normalize: bool = True) -> None:
        super().__init__(normalize)
        self.scaling_factor = scaling_factor

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(vector, matrix)
        scores = scores / math.sqrt(self.scaling_factor)
        return scores
