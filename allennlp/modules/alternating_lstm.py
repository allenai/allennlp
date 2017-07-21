
import torch
from torch.nn.utils.rnn import PackedSequence
from .augmented_lstm import AugmentedLstm


class StackedAlternatingLstm(torch.nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_layers: int,
                 recurrent_dropout_prob: float = 0.0,
                 highway: bool = True) -> None:
        super(StackedAlternatingLstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.recurrent_dropout_prob = recurrent_dropout_prob

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            direction = "forward" if layer_index % 2 == 0 else "backward"
            layer = AugmentedLstm(lstm_input_size, output_size, direction,
                                  recurrent_dropout_probability=recurrent_dropout_prob,
                                  use_highway=highway)
            lstm_input_size = output_size
            setattr(self, 'layer_{}'.format(layer_index), layer)
            layers.append(layer)
        self.lstm_layers = layers

    def forward(self, inputs: PackedSequence):  # pylint: disable=arguments-differ
        current = inputs
        for layer in self.lstm_layers:
            current = layer(current)
        return current
