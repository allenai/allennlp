# allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper

## PytorchSeq2SeqWrapper
```python
PytorchSeq2SeqWrapper(self, module:torch.nn.modules.module.Module, stateful:bool=False) -> None
```

Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
the last time step for every layer.  We just want the first one as a single output.  This
wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
set of labels.  The linear layer needs to know its input dimension before it is called, and you
can get that from ``get_output_dim``.

In order to be wrapped with this wrapper, a class must have the following members:

    - ``self.input_size: int``
    - ``self.hidden_size: int``
    - ``def forward(inputs: PackedSequence, hidden_state: torch.Tensor) ->
      Tuple[PackedSequence, torch.Tensor]``.
    - ``self.bidirectional: bool`` (optional)

This is what pytorch's RNN's look like - just make sure your class looks like those, and it
should work.

Note that we *require* you to pass a binary mask of shape (batch_size, sequence_length)
when you call this module, to avoid subtle bugs around masking.  If you already have a
``PackedSequence`` you can pass ``None`` as the second parameter.

We support stateful RNNs where the final state from each batch is used as the initial
state for the subsequent batch by passing ``stateful=True`` to the constructor.

### get_input_dim
```python
PytorchSeq2SeqWrapper.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
PytorchSeq2SeqWrapper.get_output_dim(self) -> int
```

Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
This is `not` the shape of the returned tensor, but the last element of that shape.

### is_bidirectional
```python
PytorchSeq2SeqWrapper.is_bidirectional(self) -> bool
```

Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.

### forward
```python
PytorchSeq2SeqWrapper.forward(self, inputs:torch.Tensor, mask:torch.Tensor, hidden_state:torch.Tensor=None) -> torch.Tensor
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

