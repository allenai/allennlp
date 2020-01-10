# allennlp.modules.seq2seq_encoders.multi_head_self_attention

## MultiHeadSelfAttention
```python
MultiHeadSelfAttention(self, num_heads:int, input_dim:int, attention_dim:int, values_dim:int, output_projection_dim:int=None, attention_dropout_prob:float=0.1) -> None
```

This class implements the key-value scaled dot product attention mechanism
detailed in the paper `Attention is all you Need
<https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

The attention mechanism is a weighted sum of a projection V of the inputs, with respect
to the scaled, normalised dot product of Q and K, which are also both linear projections
of the input. This procedure is repeated for each attention head, using different parameters.

Parameters
----------
num_heads : ``int``, required.
    The number of attention heads to use.
input_dim : ``int``, required.
    The size of the last dimension of the input tensor.
attention_dim ``int``, required.
    The total dimension of the query and key projections which comprise the
    dot product attention function. Must be divisible by ``num_heads``.
values_dim : ``int``, required.
    The total dimension which the input is projected to for representing the values,
    which are combined using the attention. Must be divisible by ``num_heads``.
output_projection_dim : ``int``, optional (default = None)
    The dimensionality of the final output projection. If this is not passed
    explicitly, the projection has size `input_size`.
attention_dropout_prob : ``float``, optional (default = 0.1).
    The dropout probability applied to the normalised attention
    distributions.

### is_bidirectional
```python
MultiHeadSelfAttention.is_bidirectional(self)
```

Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.

### forward
```python
MultiHeadSelfAttention.forward(self, inputs:torch.Tensor, mask:torch.LongTensor=None) -> torch.FloatTensor
```

Parameters
----------
inputs : ``torch.FloatTensor``, required.
    A tensor of shape (batch_size, timesteps, input_dim)
mask : ``torch.FloatTensor``, optional (default = None).
    A tensor of shape (batch_size, timesteps).

Returns
-------
A tensor of shape (batch_size, timesteps, output_projection_dim),
where output_projection_dim = input_dim by default.

