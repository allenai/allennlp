from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention


@Attention.register("additive")
class AdditiveAttention(Attention):
    """
    Computes attention between a vector and a matrix using an additive attention function.  This
    function has two matrices `W`, `U` and a vector `V`. The similarity between the vector
    `x` and the matrix `y` is computed as `V tanh(Wx + Uy)`.

    This attention is often referred as concat or additive attention. It was introduced in
    <https://arxiv.org/abs/1409.0473> by Bahdanau et al.

    Registered as an `Attention` with name "additive".

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    normalize : `bool`, optional (default = `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True) -> None:
        super().__init__(normalize)
        self._w_matrix = Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._v_vector = Parameter(torch.Tensor(vector_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_matrix)
        torch.nn.init.xavier_uniform_(self._u_matrix)
        torch.nn.init.xavier_uniform_(self._v_vector)

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        intermediate = torch.tanh(intermediate)
        return intermediate.matmul(self._v_vector).squeeze(2)
