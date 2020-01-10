# allennlp.modules.matrix_attention.bilinear_matrix_attention

## BilinearMatrixAttention
```python
BilinearMatrixAttention(self, matrix_1_dim:int, matrix_2_dim:int, activation:allennlp.nn.activations.Activation=None, use_input_biases:bool=False, label_dim:int=1) -> None
```

Computes attention between two matrices using a bilinear attention function.  This function has
a matrix of weights ``W`` and a bias ``b``, and the similarity between the two matrices ``X``
and ``Y`` is computed as ``X W Y^T + b``.

Parameters
----------
matrix_1_dim : ``int``, required
    The dimension of the matrix ``X``, described above.  This is ``X.size()[-1]`` - the length
    of the vector that will go into the similarity computation.  We need this so we can build
    the weight matrix correctly.
matrix_2_dim : ``int``, required
    The dimension of the matrix ``Y``, described above.  This is ``Y.size()[-1]`` - the length
    of the vector that will go into the similarity computation.  We need this so we can build
    the weight matrix correctly.
activation : ``Activation``, optional (default=linear (i.e. no activation))
    An activation function applied after the ``X W Y^T + b`` calculation.  Default is no
    activation.
use_input_biases : ``bool``, optional (default = False)
    If True, we add biases to the inputs such that the final computation
    is equivalent to the original bilinear matrix multiplication plus a
    projection of both inputs.
label_dim : ``int``, optional (default = 1)
    The number of output classes. Typically in an attention setting this will be one,
    but this parameter allows this class to function as an equivalent to ``torch.nn.Bilinear``
    for matrices, rather than vectors.

