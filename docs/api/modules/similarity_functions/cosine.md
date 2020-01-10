# allennlp.modules.similarity_functions.cosine

## CosineSimilarity
```python
CosineSimilarity(self)
```

This similarity function simply computes the cosine similarity between each pair of vectors.  It has
no parameters.

### forward
```python
CosineSimilarity.forward(self, tensor_1:torch.Tensor, tensor_2:torch.Tensor) -> torch.Tensor
```

Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.

