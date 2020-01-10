# allennlp.modules.elmo_lstm

A stacked bidirectional LSTM with skip connections between layers.

## ElmoLstm
```python
ElmoLstm(self, input_size:int, hidden_size:int, cell_size:int, num_layers:int, requires_grad:bool=False, recurrent_dropout_probability:float=0.0, memory_cell_clip_value:Union[float, NoneType]=None, state_projection_clip_value:Union[float, NoneType]=None) -> None
```

A stacked, bidirectional LSTM which uses
:class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`'s
with highway layers between the inputs to layers.
The inputs to the forward and backward directions are independent - forward and backward
states are not concatenated between layers.

Additionally, this LSTM maintains its `own` state, which is updated every time
``forward`` is called. It is dynamically resized for different batch sizes and is
designed for use with non-continuous inputs (i.e inputs which aren't formatted as a stream,
such as text used for a language modeling task, which is how stateful RNNs are typically used).
This is non-standard, but can be thought of as having an "end of sentence" state, which is
carried across different sentences.

Parameters
----------
input_size : ``int``, required
    The dimension of the inputs to the LSTM.
hidden_size : ``int``, required
    The dimension of the outputs of the LSTM.
cell_size : ``int``, required.
    The dimension of the memory cell of the
    :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`.
num_layers : ``int``, required
    The number of bidirectional LSTMs to use.
requires_grad : ``bool``, optional
    If True, compute gradient of ELMo parameters for fine tuning.
recurrent_dropout_probability : ``float``, optional (default = 0.0)
    The dropout probability to be used in a dropout scheme as stated in
    `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
    <https://arxiv.org/abs/1512.05287>`_ .
state_projection_clip_value : ``float``, optional, (default = None)
    The magnitude with which to clip the hidden_state after projecting it.
memory_cell_clip_value : ``float``, optional, (default = None)
    The magnitude with which to clip the memory cell.

### forward
```python
ElmoLstm.forward(self, inputs:torch.Tensor, mask:torch.LongTensor) -> torch.Tensor
```

Parameters
----------
inputs : ``torch.Tensor``, required.
    A Tensor of shape ``(batch_size, sequence_length, hidden_size)``.
mask : ``torch.LongTensor``, required.
    A binary mask of shape ``(batch_size, sequence_length)`` representing the
    non-padded elements in each sequence in the batch.

Returns
-------
A ``torch.Tensor`` of shape (num_layers, batch_size, sequence_length, hidden_size),
where the num_layers dimension represents the LSTM output from that layer.

### load_weights
```python
ElmoLstm.load_weights(self, weight_file:str) -> None
```

Load the pre-trained weights from the file.

