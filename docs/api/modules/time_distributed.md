# allennlp.modules.time_distributed

A wrapper that unrolls the second (time) dimension of a tensor
into the first (batch) dimension, applies some other ``Module``,
and then rolls the time dimension back up.

## TimeDistributed
```python
TimeDistributed(self, module)
```

Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.

Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
``batch_size`` is second - we always just combine the first two dimensions, then split them.

It also reshapes keyword arguments unless they are not tensors or their name is specified in
the optional ``pass_through`` iterable.

### forward
```python
TimeDistributed.forward(self, *inputs, pass_through:List[str]=None, **kwargs)
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

