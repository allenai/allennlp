# allennlp.modules.seq2seq_decoders.lstm_cell_decoder_net

## LstmCellDecoderNet
```python
LstmCellDecoderNet(self, decoding_dim:int, target_embedding_dim:int, attention:Union[allennlp.modules.attention.attention.Attention, NoneType]=None, bidirectional_input:bool=False) -> None
```

This decoder net implements simple decoding network with LSTMCell and Attention.

Parameters
----------
decoding_dim : ``int``, required
    Defines dimensionality of output vectors.
target_embedding_dim : ``int``, required
    Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
    as input of following step, this is also an input dimensionality.
attention : ``Attention``, optional (default = None)
    If you want to use attention to get a dynamic summary of the encoder outputs at each step
    of decoding, this is the function used to compute similarity between the decoder hidden
    state and encoder outputs.

### forward
```python
LstmCellDecoderNet.forward(self, previous_state:Dict[str, torch.Tensor], encoder_outputs:torch.Tensor, source_mask:torch.Tensor, previous_steps_predictions:torch.Tensor, previous_steps_mask:Union[torch.Tensor, NoneType]=None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]
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

