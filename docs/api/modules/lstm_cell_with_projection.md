# allennlp.modules.lstm_cell_with_projection

An LSTM with Recurrent Dropout, a hidden_state which is projected and
clipping on both the hidden state and the memory state of the LSTM.

## LstmCellWithProjection
```python
LstmCellWithProjection(self, input_size:int, hidden_size:int, cell_size:int, go_forward:bool=True, recurrent_dropout_probability:float=0.0, memory_cell_clip_value:Union[float, NoneType]=None, state_projection_clip_value:Union[float, NoneType]=None) -> None
```

An LSTM with Recurrent Dropout and a projected and clipped hidden state and
memory. Note: this implementation is slower than the native Pytorch LSTM because
it cannot make use of CUDNN optimizations for stacked RNNs due to and
variational dropout and the custom nature of the cell state.

Parameters
----------
input_size : ``int``, required.
    The dimension of the inputs to the LSTM.
hidden_size : ``int``, required.
    The dimension of the outputs of the LSTM.
cell_size : ``int``, required.
    The dimension of the memory cell used for the LSTM.
go_forward : ``bool``, optional (default = True)
    The direction in which the LSTM is applied to the sequence.
    Forwards by default, or backwards if False.
recurrent_dropout_probability : ``float``, optional (default = 0.0)
    The dropout probability to be used in a dropout scheme as stated in
    `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
    <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
    applies a fixed dropout mask per sequence to the recurrent connection of the
    LSTM.
state_projection_clip_value : ``float``, optional, (default = None)
    The magnitude with which to clip the hidden_state after projecting it.
memory_cell_clip_value : ``float``, optional, (default = None)
    The magnitude with which to clip the memory cell.

Returns
-------
output_accumulator : ``torch.FloatTensor``
    The outputs of the LSTM for each timestep. A tensor of shape
    (batch_size, max_timesteps, hidden_size) where for a given batch
    element, all outputs past the sequence length for that batch are
    zero tensors.
final_state : ``Tuple[torch.FloatTensor, torch.FloatTensor]``
    The final (state, memory) states of the LSTM, with shape
    (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
    respectively. The first dimension is 1 in order to match the Pytorch
    API for returning stacked LSTM states.

### forward
```python
LstmCellWithProjection.forward(self, inputs:torch.FloatTensor, batch_lengths:List[int], initial_state:Union[Tuple[torch.Tensor, torch.Tensor], NoneType]=None)
```

Parameters
----------
inputs : ``torch.FloatTensor``, required.
    A tensor of shape (batch_size, num_timesteps, input_size)
    to apply the LSTM over.
batch_lengths : ``List[int]``, required.
    A list of length batch_size containing the lengths of the sequences in batch.
initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
    A tuple (state, memory) representing the initial hidden state and memory
    of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
    ``memory`` has shape (1, batch_size, cell_size).

Returns
-------
output_accumulator : ``torch.FloatTensor``
    The outputs of the LSTM for each timestep. A tensor of shape
    (batch_size, max_timesteps, hidden_size) where for a given batch
    element, all outputs past the sequence length for that batch are
    zero tensors.
final_state : ``Tuple[``torch.FloatTensor, torch.FloatTensor]``
    A tuple (state, memory) representing the initial hidden state and memory
    of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
    ``memory`` has shape (1, batch_size, cell_size).

