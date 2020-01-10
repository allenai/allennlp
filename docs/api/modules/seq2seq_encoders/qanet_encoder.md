# allennlp.modules.seq2seq_encoders.qanet_encoder

## QaNetEncoder
```python
QaNetEncoder(self, input_dim:int, hidden_dim:int, attention_projection_dim:int, feedforward_hidden_dim:int, num_blocks:int, num_convs_per_block:int, conv_kernel_size:int, num_attention_heads:int, use_positional_encoding:bool=True, dropout_prob:float=0.1, layer_dropout_undecayed_prob:float=0.1, attention_dropout_prob:float=0) -> None
```

Stack multiple QANetEncoderBlock into one sequence encoder.

Parameters
----------
input_dim : ``int``, required.
    The input dimension of the encoder.
hidden_dim : ``int``, required.
    The hidden dimension used for convolution output channels, multi-head attention output
    and the final output of feedforward layer.
attention_projection_dim : ``int``, required.
    The dimension of the linear projections for the self-attention layers.
feedforward_hidden_dim : ``int``, required.
    The middle dimension of the FeedForward network. The input and output
    dimensions are fixed to ensure sizes match up for the self attention layers.
num_blocks : ``int``, required.
    The number of stacked encoder blocks.
num_convs_per_block : ``int``, required.
    The number of convolutions in each block.
conv_kernel_size : ``int``, required.
    The kernel size for convolution.
num_attention_heads : ``int``, required.
    The number of attention heads to use per layer.
use_positional_encoding : ``bool``, optional, (default = True)
    Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
    as without this feature, the self attention layers have no idea of absolute or relative
    position (as they are just computing pairwise similarity between vectors of elements),
    which can be important features for many tasks.
dropout_prob : ``float``, optional, (default = 0.1)
    The dropout probability for the feedforward network.
layer_dropout_undecayed_prob : ``float``, optional, (default = 0.1)
    The initial dropout probability for layer dropout, and this might decay w.r.t the depth
    of the layer. For each mini-batch, the convolution/attention/ffn sublayer is
    stochastically dropped according to its layer dropout probability.
attention_dropout_prob : ``float``, optional, (default = 0)
    The dropout probability for the attention distributions in the attention layer.

### get_input_dim
```python
QaNetEncoder.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
QaNetEncoder.get_output_dim(self) -> int
```

Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
This is `not` the shape of the returned tensor, but the last element of that shape.

### is_bidirectional
```python
QaNetEncoder.is_bidirectional(self) -> bool
```

Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.

### forward
```python
QaNetEncoder.forward(self, inputs:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

## QaNetEncoderBlock
```python
QaNetEncoderBlock(self, input_dim:int, hidden_dim:int, attention_projection_dim:int, feedforward_hidden_dim:int, num_convs:int, conv_kernel_size:int, num_attention_heads:int, use_positional_encoding:bool=True, dropout_prob:float=0.1, layer_dropout_undecayed_prob:float=0.1, attention_dropout_prob:float=0) -> None
```

Implements the encoder block described in `QANet: Combining Local Convolution with Global
Self-attention for Reading Comprehension <https://openreview.net/forum?id=B14TlG-RW>`_ .

One encoder block mainly contains 4 parts:

    1. Add position embedding.
    2. Several depthwise seperable convolutions.
    3. Multi-headed self attention, which uses 2 learnt linear projections
       to perform a dot-product similarity between every pair of elements
       scaled by the square root of the sequence length.
    4. A two-layer FeedForward network.

Parameters
----------
input_dim : ``int``, required.
    The input dimension of the encoder.
hidden_dim : ``int``, required.
    The hidden dimension used for convolution output channels, multi-head attention output
    and the final output of feedforward layer.
attention_projection_dim : ``int``, required.
    The dimension of the linear projections for the self-attention layers.
feedforward_hidden_dim : ``int``, required.
    The middle dimension of the FeedForward network. The input and output
    dimensions are fixed to ensure sizes match up for the self attention layers.
num_convs : ``int``, required.
    The number of convolutions in each block.
conv_kernel_size : ``int``, required.
    The kernel size for convolution.
num_attention_heads : ``int``, required.
    The number of attention heads to use per layer.
use_positional_encoding : ``bool``, optional, (default = True)
    Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
    as without this feature, the self attention layers have no idea of absolute or relative
    position (as they are just computing pairwise similarity between vectors of elements),
    which can be important features for many tasks.
dropout_prob : ``float``, optional, (default = 0.1)
    The dropout probability for the feedforward network.
layer_dropout_undecayed_prob : ``float``, optional, (default = 0.1)
    The initial dropout probability for layer dropout, and this might decay w.r.t the depth
    of the layer. For each mini-batch, the convolution/attention/ffn sublayer is randomly
    dropped according to its layer dropout probability.
attention_dropout_prob : ``float``, optional, (default = 0)
    The dropout probability for the attention distributions in the attention layer.

### get_input_dim
```python
QaNetEncoderBlock.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
QaNetEncoderBlock.get_output_dim(self) -> int
```

Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
This is `not` the shape of the returned tensor, but the last element of that shape.

### is_bidirectional
```python
QaNetEncoderBlock.is_bidirectional(self)
```

Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.

### forward
```python
QaNetEncoderBlock.forward(self, inputs:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

