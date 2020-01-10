# allennlp.modules.seq2seq_encoders.compose_encoder

## ComposeEncoder
```python
ComposeEncoder(self, encoders:List[allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder])
```
This class can be used to compose several encoders in sequence.

Among other things, this can be used to add a "pre-contextualizer" before a Seq2SeqEncoder.

Parameters
----------
encoders : ``List[Seq2SeqEncoder]``, required.
    A non-empty list of encoders to compose. The encoders must match in bidirectionality.

### forward
```python
ComposeEncoder.forward(self, inputs:torch.Tensor, mask:torch.LongTensor=None) -> torch.Tensor
```

Parameters
----------
inputs : ``torch.Tensor``, required.
    A tensor of shape (batch_size, timesteps, input_dim)
mask : ``torch.LongTensor``, optional (default = None).
    A tensor of shape (batch_size, timesteps).

Returns
-------
A tensor computed by composing the sequence of encoders.

### get_input_dim
```python
ComposeEncoder.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
ComposeEncoder.get_output_dim(self) -> int
```

Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
This is `not` the shape of the returned tensor, but the last element of that shape.

### is_bidirectional
```python
ComposeEncoder.is_bidirectional(self) -> bool
```

Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.

