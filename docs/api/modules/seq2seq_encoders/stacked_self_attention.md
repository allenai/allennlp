# allennlp.modules.seq2seq_encoders.stacked_self_attention

## StackedSelfAttentionEncoder
```python
StackedSelfAttentionEncoder(self, input_dim:int, hidden_dim:int, projection_dim:int, feedforward_hidden_dim:int, num_layers:int, num_attention_heads:int, use_positional_encoding:bool=True, dropout_prob:float=0.1, residual_dropout_prob:float=0.2, attention_dropout_prob:float=0.1) -> None
```

Implements a stacked self-attention encoder similar to the Transformer
architecture in `Attention is all you Need
<https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

This encoder combines 3 layers in a 'block':

1. A 2 layer FeedForward network.
2. Multi-headed self attention, which uses 2 learnt linear projections
   to perform a dot-product similarity between every pair of elements
   scaled by the square root of the sequence length.
3. Layer Normalisation.

These are then stacked into ``num_layers`` layers.

Parameters
----------
input_dim : ``int``, required.
    The input dimension of the encoder.
hidden_dim : ``int``, required.
    The hidden dimension used for the _input_ to self attention layers
    and the _output_ from the feedforward layers.
projection_dim : ``int``, required.
    The dimension of the linear projections for the self-attention layers.
feedforward_hidden_dim : ``int``, required.
    The middle dimension of the FeedForward network. The input and output
    dimensions are fixed to ensure sizes match up for the self attention layers.
num_layers : ``int``, required.
    The number of stacked self attention -> feedfoward -> layer normalisation blocks.
num_attention_heads : ``int``, required.
    The number of attention heads to use per layer.
use_positional_encoding : ``bool``, optional, (default = True)
    Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
    as without this feature, the self attention layers have no idea of absolute or relative
    position (as they are just computing pairwise similarity between vectors of elements),
    which can be important features for many tasks.
dropout_prob : ``float``, optional, (default = 0.1)
    The dropout probability for the feedforward network.
residual_dropout_prob : ``float``, optional, (default = 0.2)
    The dropout probability for the residual connections.
attention_dropout_prob : ``float``, optional, (default = 0.1)
    The dropout probability for the attention distributions in each attention layer.

### get_input_dim
```python
StackedSelfAttentionEncoder.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
StackedSelfAttentionEncoder.get_output_dim(self) -> int
```

Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
This is `not` the shape of the returned tensor, but the last element of that shape.

### is_bidirectional
```python
StackedSelfAttentionEncoder.is_bidirectional(self)
```

Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.

### forward
```python
StackedSelfAttentionEncoder.forward(self, inputs:torch.Tensor, mask:torch.Tensor)
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

