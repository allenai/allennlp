# allennlp.modules.seq2seq_encoders.feedforward_encoder

## FeedForwardEncoder
```python
FeedForwardEncoder(self, feedforward:allennlp.modules.feedforward.FeedForward) -> None
```

This class applies the `FeedForward` to each item in sequences.

### get_input_dim
```python
FeedForwardEncoder.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
FeedForwardEncoder.get_output_dim(self) -> int
```

Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
This is `not` the shape of the returned tensor, but the last element of that shape.

### is_bidirectional
```python
FeedForwardEncoder.is_bidirectional(self) -> bool
```

Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.

### forward
```python
FeedForwardEncoder.forward(self, inputs:torch.Tensor, mask:torch.LongTensor=None) -> torch.Tensor
```

Parameters
----------
inputs : ``torch.Tensor``, required.
    A tensor of shape (batch_size, timesteps, input_dim)
mask : ``torch.LongTensor``, optional (default = None).
    A tensor of shape (batch_size, timesteps).

Returns
-------
A tensor of shape (batch_size, timesteps, output_dim).

