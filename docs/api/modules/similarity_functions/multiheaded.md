# allennlp.modules.similarity_functions.multiheaded

## MultiHeadedSimilarity
```python
MultiHeadedSimilarity(self, num_heads:int, tensor_1_dim:int, tensor_1_projected_dim:int=None, tensor_2_dim:int=None, tensor_2_projected_dim:int=None, internal_similarity:allennlp.modules.similarity_functions.similarity_function.SimilarityFunction=DotProductSimilarity()) -> None
```

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

### forward
```python
MultiHeadedSimilarity.forward(self, tensor_1:torch.Tensor, tensor_2:torch.Tensor) -> torch.Tensor
```

Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.

