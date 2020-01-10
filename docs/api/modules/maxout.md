# allennlp.modules.maxout

A maxout neural network.

## Maxout
```python
Maxout(self, input_dim:int, num_layers:int, output_dims:Union[int, Sequence[int]], pool_sizes:Union[int, Sequence[int]], dropout:Union[float, Sequence[float]]=0.0) -> None
```

This ``Module`` is a maxout neural network.

Parameters
----------
input_dim : ``int``, required
    The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
num_layers : ``int``, required
    The number of maxout layers to apply to the input.
output_dims : ``Union[int, Sequence[int]]``, required
    The output dimension of each of the maxout layers.  If this is a single ``int``, we use
    it for all maxout layers.  If it is a ``Sequence[int]``, ``len(output_dims)`` must be
    ``num_layers``.
pool_sizes : ``Union[int, Sequence[int]]``, required
    The size of max-pools.  If this is a single ``int``, we use
    it for all maxout layers.  If it is a ``Sequence[int]``, ``len(pool_sizes)`` must be
    ``num_layers``.
dropout : ``Union[float, Sequence[float]]``, optional (default = 0.0)
    If given, we will apply this amount of dropout after each layer.  Semantics of ``float``
    versus ``Sequence[float]`` is the same as with other parameters.

