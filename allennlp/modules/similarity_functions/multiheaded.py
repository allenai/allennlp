from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.common.checks import ConfigurationError
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity


@SimilarityFunction.register("multiheaded")
class MultiHeadedSimilarity(SimilarityFunction):
    """
    This similarity function uses multiple "heads" to compute similarity.  That is, we take the
    input tensors and project them into a number of new tensors, and compute similarities on each
    of the projected tensors individually.  The result here has one more dimension than a typical
    similarity function.

    For example, say we have two input tensors, both of shape ``(batch_size, sequence_length,
    100)``, and that we want 5 similarity heads.  We'll project these tensors with a ``100x100``
    matrix, then split the resultant tensors to have shape ``(batch_size, sequence_length, 5,
    20)``.  Then we call a wrapped similarity function on the result (by default just a dot
    product), giving a tensor of shape ``(batch_size, sequence_length, 5)``.

    Parameters
    ----------
    num_heads : ``int``
        The number of similarity heads to compute.
    tensor_1_dim : ``int``
        The dimension of the first tensor described above.  This is ``tensor.size()[-1]`` - the
        length of the vector `before` the multi-headed projection.  We need this so we can build
        the weight matrix correctly.
    tensor_1_projected_dim : ``int``, optional
        The dimension of the first tensor `after` the multi-headed projection, `before` we split
        into multiple heads.  This number must be divisible evenly by ``num_heads``.  If not given,
        we default to ``tensor_1_dim``.
    tensor_2_dim : ``int``, optional
        The dimension of the second tensor described above.  This is ``tensor.size()[-1]`` - the
        length of the vector `before` the multi-headed projection.  We need this so we can build
        the weight matrix correctly.  If not given, we default to ``tensor_1_dim``.
    tensor_2_projected_dim : ``int``, optional
        The dimension of the second tensor `after` the multi-headed projection, `before` we split
        into multiple heads.  This number must be divisible evenly by ``num_heads``.  If not given,
        we default to ``tensor_2_dim``.
    internal_similarity : ``SimilarityFunction``, optional
        The ``SimilarityFunction`` to call on the projected, multi-headed tensors.  The default is
        to use a dot product.
    """
    def __init__(self,
                 num_heads: int,
                 tensor_1_dim: int,
                 tensor_1_projected_dim: int = None,
                 tensor_2_dim: int = None,
                 tensor_2_projected_dim: int = None,
                 internal_similarity: SimilarityFunction = DotProductSimilarity()) -> None:
        super(MultiHeadedSimilarity, self).__init__()
        self.num_heads = num_heads
        self._internal_similarity = internal_similarity
        tensor_1_projected_dim = tensor_1_projected_dim or tensor_1_dim
        tensor_2_dim = tensor_2_dim or tensor_1_dim
        tensor_2_projected_dim = tensor_2_projected_dim or tensor_2_dim
        if tensor_1_projected_dim % num_heads != 0:
            raise ConfigurationError("Projected dimension not divisible by number of heads: %d, %d"
                                     % (tensor_1_projected_dim, num_heads))
        if tensor_2_projected_dim % num_heads != 0:
            raise ConfigurationError("Projected dimension not divisible by number of heads: %d, %d"
                                     % (tensor_2_projected_dim, num_heads))
        self._tensor_1_projection = Parameter(torch.Tensor(tensor_1_dim, tensor_1_projected_dim))
        self._tensor_2_projection = Parameter(torch.Tensor(tensor_2_dim, tensor_2_projected_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._tensor_1_projection)
        torch.nn.init.xavier_uniform_(self._tensor_2_projection)

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        projected_tensor_1 = torch.matmul(tensor_1, self._tensor_1_projection)
        projected_tensor_2 = torch.matmul(tensor_2, self._tensor_2_projection)

        # Here we split the last dimension of the tensors from (..., projected_dim) to
        # (..., num_heads, projected_dim / num_heads), using tensor.view().
        last_dim_size = projected_tensor_1.size(-1) // self.num_heads
        new_shape = list(projected_tensor_1.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_1 = projected_tensor_1.view(*new_shape)
        last_dim_size = projected_tensor_2.size(-1) // self.num_heads
        new_shape = list(projected_tensor_2.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_2 = projected_tensor_2.view(*new_shape)

        # And then we pass this off to our internal similarity function.  Because the similarity
        # functions don't care what dimension their input has, and only look at the last dimension,
        # we don't need to do anything special here.  It will just compute similarity on the
        # projection dimension for each head, returning a tensor of shape (..., num_heads).
        return self._internal_similarity(split_tensor_1, split_tensor_2)
