
import torch
from torch.nn.modules.rnn import RNNCellBase


class AlternatingLSTM(RNNCellBase):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_layers: int,
                 recurrent_dropout_prob: float = 0.0,
                 highway: bool = True):
        super(AlternatingLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.out_dim = output_size
        self.recurrent_dropout_prob = recurrent_dropout_prob

        layers = []
        for layer_index in range(num_layers):
            if layer_index == 0:
                input_size = input_size
            else:
                input_size = output_size
            if layer_index % 2 == 0:
                direction = 1
            else:
                direction = -1
            layer = AugmentedLstmLayer(input_size, output_size, direction,
                                       recurrent_dropout_prob=recurrent_dropout_prob, highway=highway)
            setattr(self, 'layer_{}'.format(layer_index), layer)
            layers.append(layer)
        self.layers = layers

    def forward(self, input):
        current = input
        for layer in self.layers:
            current = layer(current)
        return current


class RecurrentHighwayLayer(torch.nn.Module):

    def __init__(self, input_size, state_size):

        super(RecurrentHighwayLayer, self).__init__()

        self.linear = torch.nn.Linear(input_size + state_size, state_size)
        self.output_linear = torch.nn.Linear(input_size, state_size)

    def forward(self, layer_input: torch.Tensor, previous_layer_state: torch.Tensor):

        concatenated = torch.cat([layer_input, previous_layer_state], dim=1)
        gating_vector = torch.sigmoid(self.projection(concatenated))

        projected_input = self.output_linear(layer_input)

        return (gating_vector * previous_layer_state) + (1.0 - gating_vector) * projected_input

