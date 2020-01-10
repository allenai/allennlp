# allennlp.modules.attention.bilinear_attention

## BilinearAttention
```python
BilinearAttention(self, vector_dim:int, matrix_dim:int, activation:allennlp.nn.activations.Activation=None, normalize:bool=True) -> None
```

Computes attention between a vector and a matrix using a bilinear attention function.  This
function has a matrix of weights ``W`` and a bias ``b``, and the similarity between the vector
``x`` and the matrix ``y`` is computed as ``x^T W y + b``.

Parameters
----------
vector_dim : ``int``, required
    The dimension of the vector, ``x``, described above.  This is ``x.size()[-1]`` - the length
    of the vector that will go into the similarity computation.  We need this so we can build
    the weight matrix correctly.
matrix_dim : ``int``, required
    The dimension of the matrix, ``y``, described above.  This is ``y.size()[-1]`` - the length
    of the vector that will go into the similarity computation.  We need this so we can build
    the weight matrix correctly.
activation : ``Activation``, optional (default=linear (i.e. no activation))
    An activation function applied after the ``x^T W y + b`` calculation.  Default is no
    activation.
normalize : ``bool``, optional (default : ``True``)
    If true, we normalize the computed similarities with a softmax, to return a probability
    distribution for your attention.  If false, this is just computing a similarity score.

