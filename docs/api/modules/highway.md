# allennlp.modules.highway

A `Highway layer <https://arxiv.org/abs/1505.00387>`_ that does a gated combination of a linear
transformation and a non-linear transformation of its input.

## Highway
```python
Highway(self, input_dim:int, num_layers:int=1, activation:Callable[[torch.Tensor], torch.Tensor]=<function relu at 0x121def158>) -> None
```

A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

This module will apply a fixed number of highway layers to its input, returning the final
result.

Parameters
----------
input_dim : ``int``, required
    The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size, ...,
    input_dim)``.
num_layers : ``int``, optional (default=``1``)
    The number of highway layers to apply to the input.
activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``torch.nn.functional.relu``)
    The non-linearity to use in the highway layers.

### forward
```python
Highway.forward(self, inputs:torch.Tensor) -> torch.Tensor
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

