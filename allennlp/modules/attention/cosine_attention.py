import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.modules.attention.legacy_attention import Attention


@Attention.register("cosine")
class CosineAttention(Attention):
    """
    Computes attention between a vector and a matrix using cosine similarity.
    """
    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        a_norm = vector / (vector.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        b_norm = matrix / (matrix.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        return torch.bmm(a_norm.unsqueeze(dim=1), b_norm.transpose(-1, -2)).squeeze(1)

    @classmethod
    def from_params(cls, params: Params):
        normalize = params.pop_bool('normalize', True)
        params.assert_empty(cls.__name__)
        return CosineAttention(normalize)
