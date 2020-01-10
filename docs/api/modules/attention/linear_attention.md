# allennlp.modules.attention.linear_attention

## LinearAttention
```python
LinearAttention(self, tensor_1_dim:int, tensor_2_dim:int, combination:str='x,y', activation:allennlp.nn.activations.Activation=None, normalize:bool=True) -> None
```

This ``Attention`` module performs a dot product between a vector of weights and some
combination of the two input vectors, followed by an (optional) activation function.  The
combination used is configurable.

If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations : ``x``,
``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
elementwise.  You can list as many combinations as you want, comma separated.  For example, you
might give ``x,y,x*y`` as the ``combination`` parameter to this class.  The computed similarity
function would then be ``w^T [x; y; x*y] + b``, where ``w`` is a vector of weights, ``b`` is a
bias parameter, and ``[;]`` is vector concatenation.

Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
accomplish that with this class by using "x*y" for `combination`.

Parameters
----------
tensor_1_dim : ``int``, required
    The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
    length of the vector that will go into the similarity computation.  We need this so we can
    build weight vectors correctly.
tensor_2_dim : ``int``, required
    The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
    length of the vector that will go into the similarity computation.  We need this so we can
    build weight vectors correctly.
combination : ``str``, optional (default="x,y")
    Described above.
activation : ``Activation``, optional (default=linear (i.e. no activation))
    An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
    activation.
normalize : ``bool``, optional (default=True)

