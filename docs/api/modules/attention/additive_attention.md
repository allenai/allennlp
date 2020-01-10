# allennlp.modules.attention.additive_attention

## AdditiveAttention
```python
AdditiveAttention(self, vector_dim:int, matrix_dim:int, normalize:bool=True) -> None
```

Computes attention between a vector and a matrix using an additive attention function.  This
function has two matrices ``W``, ``U`` and a vector ``V``. The similarity between the vector
``x`` and the matrix ``y`` is computed as ``V tanh(Wx + Uy)``.

This attention is often referred as concat or additive attention. It was introduced in
<https://arxiv.org/abs/1409.0473> by Bahdanau et al.

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
normalize : ``bool``, optional (default : ``True``)
    If true, we normalize the computed similarities with a softmax, to return a probability
    distribution for your attention.  If false, this is just computing a similarity score.

