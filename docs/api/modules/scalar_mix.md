# allennlp.modules.scalar_mix

## ScalarMix
```python
ScalarMix(self, mixture_size:int, do_layer_norm:bool=False, initial_scalar_parameters:List[float]=None, trainable:bool=True) -> None
```

Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
before weighting.

### forward
```python
ScalarMix.forward(self, tensors:List[torch.Tensor], mask:torch.Tensor=None) -> torch.Tensor
```

Compute a weighted average of the ``tensors``.  The input tensors an be any shape
with at least two dimensions, but must all be the same shape.

When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

When ``do_layer_norm=False`` the ``mask`` is ignored.

