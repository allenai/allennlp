# allennlp.modules.augmented_lstm

An LSTM with Recurrent Dropout and the option to use highway
connections between layers.

## AugmentedLstm
```python
AugmentedLstm(self, input_size:int, hidden_size:int, go_forward:bool=True, recurrent_dropout_probability:float=0.0, use_highway:bool=True, use_input_projection_bias:bool=True) -> None
```

An LSTM with Recurrent Dropout and the option to use highway
connections between layers. Note: this implementation is slower
than the native Pytorch LSTM because it cannot make use of CUDNN
optimizations for stacked RNNs due to the highway layers and
variational dropout.

Parameters
----------
input_size : ``int``, required.
    The dimension of the inputs to the LSTM.
hidden_size : ``int``, required.
    The dimension of the outputs of the LSTM.
go_forward : ``bool``, optional (default = True)
    The direction in which the LSTM is applied to the sequence.
    Forwards by default, or backwards if False.
recurrent_dropout_probability : ``float``, optional (default = 0.0)
    The dropout probability to be used in a dropout scheme as stated in
    `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
    <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
    applies a fixed dropout mask per sequence to the recurrent connection of the
    LSTM. Dropout is not applied to the output sequence nor the last hidden
    state that is returned, it is only applied to all previous hidden states.
use_highway : ``bool``, optional (default = True)
    Whether or not to use highway connections between layers. This effectively involves
    reparameterising the normal output of an LSTM as::

        gate = sigmoid(W_x1 * x_t + W_h * h_t)
        output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
use_input_projection_bias : bool, optional (default = True)
    Whether or not to use a bias on the input projection layer. This is mainly here
    for backwards compatibility reasons and will be removed (and set to False)
    in future releases.

Returns
-------
output_accumulator : PackedSequence
    The outputs of the LSTM for each timestep. A tensor of shape
    (batch_size, max_timesteps, hidden_size) where for a given batch
    element, all outputs past the sequence length for that batch are
    zero tensors.

### forward
```python
AugmentedLstm.forward(self, inputs:torch.nn.utils.rnn.PackedSequence, initial_state:Union[Tuple[torch.Tensor, torch.Tensor], NoneType]=None)
```

Parameters
----------
inputs : ``PackedSequence``, required.
    A tensor of shape (batch_size, num_timesteps, input_size)
    to apply the LSTM over.

initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
    A tuple (state, memory) representing the initial hidden state and memory
    of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

Returns
-------
A PackedSequence containing a torch.FloatTensor of shape
(batch_size, num_timesteps, output_dimension) representing
the outputs of the LSTM per timestep and a tuple containing
the LSTM state, with shape (1, batch_size, hidden_size) to
match the Pytorch API.

