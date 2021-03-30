from typing import Optional, Tuple, List
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.common.checks import ConfigurationError


TensorPair = Tuple[torch.Tensor, torch.Tensor]


class StackedBidirectionalLstm(torch.nn.Module):
    """
    A standard stacked Bidirectional LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular bidirectional LSTM is the application of
    variational dropout to the hidden states and outputs of each layer apart
    from the last layer of the LSTM. Note that this will be slower, as it
    doesn't use CUDNN.

    [0]: https://arxiv.org/abs/1512.05287

    # Parameters

    input_size : `int`, required
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required
        The dimension of the outputs of the LSTM.
    num_layers : `int`, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability : `float`, optional (default = `0.0`)
        The recurrent dropout probability to be used in a dropout scheme as
        stated in [A Theoretically Grounded Application of Dropout in Recurrent
        Neural Networks][0].
    layer_dropout_probability : `float`, optional (default = `0.0`)
        The layer wise dropout probability to be used in a dropout scheme as
        stated in [A Theoretically Grounded Application of Dropout in Recurrent
        Neural Networks][0].
    use_highway : `bool`, optional (default = `True`)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        recurrent_dropout_probability: float = 0.0,
        layer_dropout_probability: float = 0.0,
        use_highway: bool = True,
    ) -> None:
        super().__init__()

        # Required to be wrapped with a `PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):

            forward_layer = AugmentedLstm(
                lstm_input_size,
                hidden_size,
                go_forward=True,
                recurrent_dropout_probability=recurrent_dropout_probability,
                use_highway=use_highway,
                use_input_projection_bias=False,
            )
            backward_layer = AugmentedLstm(
                lstm_input_size,
                hidden_size,
                go_forward=False,
                recurrent_dropout_probability=recurrent_dropout_probability,
                use_highway=use_highway,
                use_input_projection_bias=False,
            )

            lstm_input_size = hidden_size * 2
            self.add_module("forward_layer_{}".format(layer_index), forward_layer)
            self.add_module("backward_layer_{}".format(layer_index), backward_layer)
            layers.append([forward_layer, backward_layer])
        self.lstm_layers = layers
        self.layer_dropout = InputVariationalDropout(layer_dropout_probability)

    def forward(
        self, inputs: PackedSequence, initial_state: Optional[TensorPair] = None
    ) -> Tuple[PackedSequence, TensorPair]:
        """
        # Parameters

        inputs : `PackedSequence`, required.
            A batch first `PackedSequence` to run the stacked LSTM over.
        initial_state : `Tuple[torch.Tensor, torch.Tensor]`, optional, (default = `None`)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (num_layers, batch_size, output_dimension * 2).

        # Returns

        output_sequence : `PackedSequence`
            The encoded sequence of shape (batch_size, sequence_length, hidden_size * 2)
        final_states: `torch.Tensor`
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers * 2, batch_size, hidden_size * 2).
        """
        if initial_state is None:
            hidden_states: List[Optional[TensorPair]] = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        output_sequence = inputs
        final_h = []
        final_c = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, "forward_layer_{}".format(i))
            backward_layer = getattr(self, "backward_layer_{}".format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(output_sequence, state)
            backward_output, final_backward_state = backward_layer(output_sequence, state)

            forward_output, lengths = pad_packed_sequence(forward_output, batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output, batch_first=True)

            output_sequence = torch.cat([forward_output, backward_output], -1)
            # Apply layer wise dropout on each output sequence apart from the
            # first (input) and last
            if i < (self.num_layers - 1):
                output_sequence = self.layer_dropout(output_sequence)
            output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)

            final_h.extend([final_forward_state[0], final_backward_state[0]])
            final_c.extend([final_forward_state[1], final_backward_state[1]])

        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = (final_h, final_c)
        return output_sequence, final_state_tuple
