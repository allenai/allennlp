"""
A stacked LSTM with LSTM layers which alternate between going forwards over
the sequence and going backwards.
"""

from typing import Optional, Tuple, Union, List
import torch
from torch.nn.utils.rnn import PackedSequence
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.common.checks import ConfigurationError

TensorPair = Tuple[torch.Tensor, torch.Tensor]


class StackedAlternatingLstm(torch.nn.Module):
    """
    A stacked LSTM with LSTM layers which alternate between going forwards over
    the sequence and going backwards. This implementation is based on the
    description in [Deep Semantic Role Labelling - What works and what's next][0].

    [0]: https://www.aclweb.org/anthology/P17-1044.pdf
    [1]: https://arxiv.org/abs/1512.05287

    # Parameters

    input_size : `int`, required
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required
        The dimension of the outputs of the LSTM.
    num_layers : `int`, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability : `float`, optional (default = `0.0`)
        The dropout probability to be used in a dropout scheme as stated in
        [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks][1].
    use_input_projection_bias : `bool`, optional (default = `True`)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.

    # Returns

    output_accumulator : `PackedSequence`
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        recurrent_dropout_probability: float = 0.0,
        use_highway: bool = True,
        use_input_projection_bias: bool = True,
    ) -> None:
        super().__init__()

        # Required to be wrapped with a `PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            go_forward = layer_index % 2 == 0
            layer = AugmentedLstm(
                lstm_input_size,
                hidden_size,
                go_forward,
                recurrent_dropout_probability=recurrent_dropout_probability,
                use_highway=use_highway,
                use_input_projection_bias=use_input_projection_bias,
            )
            lstm_input_size = hidden_size
            self.add_module("layer_{}".format(layer_index), layer)
            layers.append(layer)
        self.lstm_layers = layers

    def forward(
        self, inputs: PackedSequence, initial_state: Optional[TensorPair] = None
    ) -> Tuple[Union[torch.Tensor, PackedSequence], TensorPair]:
        """
        # Parameters

        inputs : `PackedSequence`, required.
            A batch first `PackedSequence` to run the stacked LSTM over.
        initial_state : `Tuple[torch.Tensor, torch.Tensor]`, optional, (default = `None`)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        # Returns

        output_sequence : `PackedSequence`
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: `Tuple[torch.Tensor, torch.Tensor]`
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        if not initial_state:
            hidden_states: List[Optional[TensorPair]] = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        output_sequence = inputs
        final_states = []
        for i, state in enumerate(hidden_states):
            layer = getattr(self, "layer_{}".format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            output_sequence, final_state = layer(output_sequence, state)
            final_states.append(final_state)

        final_hidden_state, final_cell_state = tuple(
            torch.cat(state_list, 0) for state_list in zip(*final_states)
        )
        return output_sequence, (final_hidden_state, final_cell_state)
