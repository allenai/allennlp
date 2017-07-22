
import torch
from torch.nn.utils.rnn import PackedSequence
from allennlp.modules.augmented_lstm import AugmentedLstm


class StackedAlternatingLstm(torch.nn.Module):
    """
    A stacked LSTM with LSTM layers which alternate between going forwards over
    the sequence and going backwards. This implementation is based on the
    description in `Deep Semantic Role Labelling - What works and what's next
    <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    output_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .

    Return
    ------
    output_accumulator : PackedSequence
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, output_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_layers: int,
                 recurrent_dropout_probability: float = 0.0,
                 use_highway: bool = True) -> None:
        super(StackedAlternatingLstm, self).__init__()

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            direction = "forward" if layer_index % 2 == 0 else "backward"
            layer = AugmentedLstm(lstm_input_size, output_size, direction,
                                  recurrent_dropout_probability=recurrent_dropout_probability,
                                  use_highway=use_highway)
            lstm_input_size = output_size
            setattr(self, 'layer_{}'.format(layer_index), layer)
            layers.append(layer)
        self.lstm_layers = layers

    def forward(self, inputs: PackedSequence):  # pylint: disable=arguments-differ
        current = inputs
        for layer in self.lstm_layers:
            current = layer(current)
        return current
