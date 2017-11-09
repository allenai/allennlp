"""
A stacked bidirectional LSTM with skip connections between layers.
"""

from typing import Optional, Tuple
import torch
from torch.nn.utils.rnn import PackedSequence
from allennlp.modules.elmo_lstm_cell import ElmoLstmCell
from allennlp.common.checks import ConfigurationError


class ElmoLstm(torch.nn.Module):
    """
    A stacked, bidirectional LSTM with highway layers between the inputs to layers.
    The inputs to the forward and backward directions are independent.

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    cell_size : int, required.
        The cell dimension of the :class:~`allennlp.modules.elmo_lstm_cell.ElmoLstmCell`.
    num_layers : int, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    state_projection_clip_value: float, optional, (default = 3)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: float, optional, (default = 3)
        The magnitude with which to clip the memory cell.

    Returns
    -------
    output_accumulator : PackedSequence
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 num_layers: int,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: float = 3.0,
                 state_projection_clip_value: float = 3.0) -> None:
        super(ElmoLstm, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = ElmoLstmCell(lstm_input_size,
                                         cell_size,
                                         hidden_size,
                                         go_forward,
                                         recurrent_dropout_probability,
                                         memory_cell_clip_value,
                                         state_projection_clip_value)
            backward_layer = ElmoLstmCell(lstm_input_size,
                                          cell_size,
                                          hidden_size,
                                          not go_forward,
                                          recurrent_dropout_probability,
                                          memory_cell_clip_value,
                                          state_projection_clip_value)

            lstm_input_size = hidden_size
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
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
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, 2 * hidden_size).
        """
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))

        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, (forward_layer, backward_layer, state) in enumerate(zip(self.forward_layers,
                                                                                 self.backward_layers,
                                                                                 hidden_states)):
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence, state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence, state)

            # Vanilla highway layers.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence,
                                               backward_output_sequence], -1))

            final_states.append(torch.cat([forward_state, backward_state], -1))

        # TODO(Mark): figure out the best api to return these.
        sequence_outputs = torch.stack(sequence_outputs)
        final_state_tuple = (torch.cat(state_list, 0) for state_list in zip(*final_states))
        return sequence_outputs, final_state_tuple
