# allennlp.modules.seq2seq_decoders.stacked_self_attention_decoder_net

## StackedSelfAttentionDecoderNet
```python
StackedSelfAttentionDecoderNet(self, decoding_dim:int, target_embedding_dim:int, feedforward_hidden_dim:int, num_layers:int, num_attention_heads:int, use_positional_encoding:bool=True, positional_encoding_max_steps:int=5000, dropout_prob:float=0.1, residual_dropout_prob:float=0.2, attention_dropout_prob:float=0.1) -> None
```

A Stacked self-attention decoder implementation.

Parameters
----------
decoding_dim : ``int``, required
    Defines dimensionality of output vectors.
target_embedding_dim : ``int``, required
    Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
    as input of following step, this is also an input dimensionality.
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

### init_decoder_state
```python
StackedSelfAttentionDecoderNet.init_decoder_state(self, encoder_out:Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]
```

Initialize the encoded state to be passed to the first decoding time step.

Parameters
----------
batch_size : ``int``
    Size of batch
final_encoder_output : ``torch.Tensor``
    Last state of the Encoder

Returns
-------
``Dict[str, torch.Tensor]``
Initial state

### forward
```python
StackedSelfAttentionDecoderNet.forward(self, previous_state:Dict[str, torch.Tensor], encoder_outputs:torch.Tensor, source_mask:torch.Tensor, previous_steps_predictions:torch.Tensor, previous_steps_mask:Union[torch.Tensor, NoneType]=None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]
```

Performs a decoding step, and returns dictionary with decoder hidden state or cache and the decoder output.
The decoder output is a 3d tensor (group_size, steps_count, decoder_output_dim)
if `self.decodes_parallel` is True, else it is a 2d tensor with (group_size, decoder_output_dim).

Parameters
----------
previous_steps_predictions : ``torch.Tensor``, required
    Embeddings of predictions on previous step.
    Shape: (group_size, steps_count, decoder_output_dim)
encoder_outputs : ``torch.Tensor``, required
    Vectors of all encoder outputs.
    Shape: (group_size, max_input_sequence_length, encoder_output_dim)
source_mask : ``torch.Tensor``, required
    This tensor contains mask for each input sequence.
    Shape: (group_size, max_input_sequence_length)
previous_state : ``Dict[str, torch.Tensor]``, required
    previous state of decoder

Returns
-------
Tuple[Dict[str, torch.Tensor], torch.Tensor]
Tuple of new decoder state and decoder output. Output should be used to generate out sequence elements

## Decoder
```python
Decoder(self, layer:torch.nn.modules.module.Module, num_layers:int) -> None
```

Transformer N layer decoder with masking.
Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html

### forward
```python
Decoder.forward(self, x:torch.Tensor, memory:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor) -> torch.Tensor
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

## DecoderLayer
```python
DecoderLayer(self, size:int, self_attn:allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer.MultiHeadedAttention, src_attn:allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer.MultiHeadedAttention, feed_forward:<module 'torch.nn.functional' from '/Users/markn/anaconda3/envs/allennlp/lib/python3.6/site-packages/torch/nn/functional.py'>, dropout:float) -> None
```

A single layer of transformer decoder.
Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html

### forward
```python
DecoderLayer.forward(self, x:torch.Tensor, memory:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor) -> torch.Tensor
```
Follow Figure 1 (right) for connections.
