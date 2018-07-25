import torch

from allennlp.common import Registrable

class SimilarityFunction(torch.nn.Module, Registrable):
    """
    A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
    function on the vectors in the last dimension.  For example, the tensors might both have shape
    `(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
    vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
    tensor of shape `(batch_size, sentence_length)`.

    The similarity function could be as simple as a dot product, or it could be a more complex,
    parameterized function.

    If you want to compute a similarity between tensors of different sizes, you need to first tile
    them in the appropriate dimensions to make them the same before you can use these functions.
    The :class:`~allennlp.modules.Attention` and :class:`~allennlp.modules.MatrixAttention` modules
    do this.
    """
    default_implementation = 'dot_product'

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.
        """
        raise NotImplementedError
