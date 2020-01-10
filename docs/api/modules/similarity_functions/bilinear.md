# allennlp.modules.similarity_functions.bilinear

## BilinearSimilarity
```python
BilinearSimilarity(self, tensor_1_dim:int, tensor_2_dim:int, activation:allennlp.nn.activations.Activation=None) -> None
```

This similarity function performs a bilinear transformation of the two input vectors.  This
function has a matrix of weights ``W`` and a bias ``b``, and the similarity between two vectors
``x`` and ``y`` is computed as ``x^T W y + b``.

Parameters
----------
tensor_1_dim : ``int``
    The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
    length of the vector that will go into the similarity computation.  We need this so we can
    build the weight matrix correctly.
tensor_2_dim : ``int``
    The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
    length of the vector that will go into the similarity computation.  We need this so we can
    build the weight matrix correctly.
activation : ``Activation``, optional (default=linear (i.e. no activation))
    An activation function applied after the ``x^T W y + b`` calculation.  Default is no
    activation.

### forward
```python
BilinearSimilarity.forward(self, tensor_1:torch.Tensor, tensor_2:torch.Tensor) -> torch.Tensor
```

Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.

