# allennlp.modules.feedforward

A feed-forward neural network.

## FeedForward
```python
FeedForward(self, input_dim:int, num_layers:int, hidden_dims:Union[int, List[int]], activations:Union[allennlp.nn.activations.Activation, List[allennlp.nn.activations.Activation]], dropout:Union[float, List[float]]=0.0) -> None
```

This ``Module`` is a feed-forward neural network, just a sequence of ``Linear`` layers with
activation functions in between.

Parameters
----------
input_dim : ``int``, required
    The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
num_layers : ``int``, required
    The number of ``Linear`` layers to apply to the input.
hidden_dims : ``Union[int, List[int]]``, required
    The output dimension of each of the ``Linear`` layers.  If this is a single ``int``, we use
    it for all ``Linear`` layers.  If it is a ``List[int]``, ``len(hidden_dims)`` must be
    ``num_layers``.
activations : ``Union[Callable, List[Callable]]``, required
    The activation function to use after each ``Linear`` layer.  If this is a single function,
    we use it after all ``Linear`` layers.  If it is a ``List[Callable]``,
    ``len(activations)`` must be ``num_layers``.
dropout : ``Union[float, List[float]]``, optional (default = 0.0)
    If given, we will apply this amount of dropout after each layer.  Semantics of ``float``
    versus ``List[float]`` is the same as with other parameters.

