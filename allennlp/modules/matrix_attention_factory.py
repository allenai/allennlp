from allennlp.modules.matrix_attention import MatrixAttention

from allennlp.common.params import Params
from allennlp.modules.legacy_matrix_attention import LegacyMatrixAttention


class MatrixAttentionFactory:
    """
             Responsible for supporting both the legacy and new attention mechanisms.
             New implementations of the attention were added because they have a much smaller memory footprint.
             The legacy attention is kept around to support backwards compatibility, although no new project should use them.

             It is assumed that they type label for memory efficient attention ends with 'matrix_attention'
             If it does not end in 'matrix_attention' we assume that it is a legacy implementation.

         """
    @classmethod
    def from_params(cls, params: Params) -> 'MatrixAttention':
        _type: str = params.get("type")
        if _type.endswith("matrix_attention"):
            return MatrixAttention.from_params(params)
        else:
            return LegacyMatrixAttention.from_params(params)