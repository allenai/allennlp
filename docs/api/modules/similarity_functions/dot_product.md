# allennlp.modules.similarity_functions.dot_product

## DotProductSimilarity
```python
DotProductSimilarity(self, scale_output:bool=False) -> None
```

This similarity function simply computes the dot product between each pair of vectors, with an
optional scaling to reduce the variance of the output elements.

Parameters
----------
scale_output : ``bool``, optional
    If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
    variance in the result.

### forward
```python
DotProductSimilarity.forward(self, tensor_1:torch.Tensor, tensor_2:torch.Tensor) -> torch.Tensor
```

Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.

