# allennlp.modules.stacked_alternating_lstm

A stacked LSTM with LSTM layers which alternate between going forwards over
the sequence and going backwards.

## StackedAlternatingLstm
```python
StackedAlternatingLstm(self, input_size:int, hidden_size:int, num_layers:int, recurrent_dropout_probability:float=0.0, use_highway:bool=True, use_input_projection_bias:bool=True) -> None
```

A stacked LSTM with LSTM layers which alternate between going forwards over
the sequence and going backwards. This implementation is based on the
description in `Deep Semantic Role Labelling - What works and what's next
<https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

Parameters
----------
input_size : ``int``, required
    The dimension of the inputs to the LSTM.
hidden_size : ``int``, required
    The dimension of the outputs of the LSTM.
num_layers : ``int``, required
    The number of stacked LSTMs to use.
recurrent_dropout_probability : ``float``, optional (default = 0.0)
    The dropout probability to be used in a dropout scheme as stated in
    `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
    <https://arxiv.org/abs/1512.05287>`_ .
use_input_projection_bias : ``bool``, optional (default = True)
    Whether or not to use a bias on the input projection layer. This is mainly here
    for backwards compatibility reasons and will be removed (and set to False)
    in future releases.

Returns
-------
output_accumulator : PackedSequence
    The outputs of the interleaved LSTMs per timestep. A tensor of shape
    (batch_size, max_timesteps, hidden_size) where for a given batch
    element, all outputs past the sequence length for that batch are
    zero tensors.

### forward
```python
StackedAlternatingLstm.forward(self, inputs:torch.nn.utils.rnn.PackedSequence, initial_state:Union[Tuple[torch.Tensor, torch.Tensor], NoneType]=None) -> Tuple[Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence], Tuple[torch.Tensor, torch.Tensor]]
```

Parameters
----------
inputs : ``PackedSequence``, required.
    A batch first ``PackedSequence`` to run the stacked LSTM over.
initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
    A tuple (state, memory) representing the initial hidden state and memory
    of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

Returns
-------
output_sequence : PackedSequence
    The encoded sequence of shape (batch_size, sequence_length, hidden_size)
final_states: Tuple[torch.Tensor, torch.Tensor]
    The per-layer final (state, memory) states of the LSTM, each with shape
    (num_layers, batch_size, hidden_size).

