# allennlp.modules.stacked_bidirectional_lstm

## StackedBidirectionalLstm
```python
StackedBidirectionalLstm(self, input_size:int, hidden_size:int, num_layers:int, recurrent_dropout_probability:float=0.0, layer_dropout_probability:float=0.0, use_highway:bool=True) -> None
```

A standard stacked Bidirectional LSTM where the LSTM layers
are concatenated between each layer. The only difference between
this and a regular bidirectional LSTM is the application of
variational dropout to the hidden states and outputs of each layer apart
from the last layer of the LSTM. Note that this will be slower, as it
doesn't use CUDNN.

Parameters
----------
input_size : ``int``, required
    The dimension of the inputs to the LSTM.
hidden_size : ``int``, required
    The dimension of the outputs of the LSTM.
num_layers : ``int``, required
    The number of stacked Bidirectional LSTMs to use.
recurrent_dropout_probability : ``float``, optional (default = 0.0)
    The recurrent dropout probability to be used in a dropout scheme as
    stated in `A Theoretically Grounded Application of Dropout in Recurrent
    Neural Networks <https://arxiv.org/abs/1512.05287>`_ .
layer_dropout_probability : ``float``, optional (default = 0.0)
    The layer wise dropout probability to be used in a dropout scheme as
    stated in  `A Theoretically Grounded Application of Dropout in
    Recurrent Neural Networks <https://arxiv.org/abs/1512.05287>`_ .
use_highway : ``bool``, optional (default = True)
    Whether or not to use highway connections between layers. This effectively involves
    reparameterising the normal output of an LSTM as::

        gate = sigmoid(W_x1 * x_t + W_h * h_t)
        output = gate * h_t  + (1 - gate) * (W_x2 * x_t)

### forward
```python
StackedBidirectionalLstm.forward(self, inputs:torch.nn.utils.rnn.PackedSequence, initial_state:Union[Tuple[torch.Tensor, torch.Tensor], NoneType]=None) -> Tuple[torch.nn.utils.rnn.PackedSequence, Tuple[torch.Tensor, torch.Tensor]]
```

Parameters
----------
inputs : ``PackedSequence``, required.
    A batch first ``PackedSequence`` to run the stacked LSTM over.
initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
    A tuple (state, memory) representing the initial hidden state and memory
    of the LSTM. Each tensor has shape (num_layers, batch_size, output_dimension * 2).

Returns
-------
output_sequence : PackedSequence
    The encoded sequence of shape (batch_size, sequence_length, hidden_size * 2)
final_states: torch.Tensor
    The per-layer final (state, memory) states of the LSTM, each with shape
    (num_layers * 2, batch_size, hidden_size * 2).

