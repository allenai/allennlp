"""
A stacked bidirectional LSTM with skip connections between layers.
"""

from typing import Optional, Tuple, List
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection
from allennlp.common.checks import ConfigurationError
from allennlp.modules.encoder_base import _EncoderBase


class ElmoLstm(_EncoderBase):
    """
    A stacked, bidirectional LSTM which uses
    :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`'s
    with highway layers between the inputs to layers.
    The inputs to the forward and backward directions are independent - forward and backward
    states are not concatenated between layers.

    Additionally, this LSTM maintains its `own` state, which is updated every time
    ``forward`` is called. It is dynamically resized for different batch sizes and is
    designed for use with non-continuous inputs. This is non-standard, but can be thought
    of as having an "end of sentence" state, which is carried across different sentences.

    Parameters
    ----------
    input_size : ``int``, required
        The dimension of the inputs to the LSTM.
    hidden_size : ``int``, required
        The dimension of the outputs of the LSTM.
    cell_size : ``int``, required.
        The dimension of the memory cell of the
        :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`.
    num_layers : ``int``, required
        The number of bidirectional LSTMs to use.
    recurrent_dropout_probability: ``float``, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    state_projection_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the memory cell.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 num_layers: int,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: Optional[float] = None,
                 state_projection_clip_value: Optional[float] = None) -> None:
        super(ElmoLstm, self).__init__(stateful=True)

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                                                   hidden_size,
                                                   cell_size,
                                                   go_forward,
                                                   recurrent_dropout_probability,
                                                   memory_cell_clip_value,
                                                   state_projection_clip_value)
            backward_layer = LstmCellWithProjection(lstm_input_size,
                                                    hidden_size,
                                                    cell_size,
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
                inputs: torch.Tensor,
                mask: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A Tensor of shape ``(batch_size, sequence_length, hidden_size)``.
        mask : ``torch.LongTensor``, required.
            A binary mask of shape ``(batch_size, sequence_length)`` representing the
            non-padded elements in each sequence in the batch.

        Returns
        -------
        A ``torch.Tensor`` of shape (num_layers, batch_size, sequence_length, hidden_size),
        where the num_layers dimension represents the LSTM output from that layer.
        """
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask)

        # stacked_sequence_output is shape (num_layers, batch_size, timesteps, encoder_dim)
        num_layers, num_valid, *_ = stacked_sequence_output.size()
        per_layer_sequence_outputs = [layer.squeeze(0) for layer in
                                      stacked_sequence_output.chunk(num_layers, 0)]

        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        if num_valid < batch_size:
            _, length, dim = per_layer_sequence_outputs[0].size()
            zeros = per_layer_sequence_outputs[0].data.new(batch_size - num_valid,
                                                           length, dim).fill_(0)
            for k in range(num_layers):
                per_layer_sequence_outputs[k] = torch.cat([per_layer_sequence_outputs[k],
                                                           zeros], 0)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                num_layers, _, state_dim = state.size()
                zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - per_layer_sequence_outputs[0].size(1)
        if sequence_length_difference > 0:
            zeros = per_layer_sequence_outputs[0].data.new(batch_size,
                                                           sequence_length_difference,
                                                           per_layer_sequence_outputs[0].size(-1)).fill_(0)
            zeros = torch.autograd.Variable(zeros)
            for k in range(num_layers):
                per_layer_sequence_outputs[k] = torch.cat([per_layer_sequence_outputs[k], zeros], 1)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)

        return torch.cat([tensor.index_select(0, restoration_indices).unsqueeze(0)
                          for tensor in per_layer_sequence_outputs], dim=0)

    def _lstm_forward(self,
                      inputs: PackedSequence,
                      initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
            (num_layers, batch_size, 2 * cell_size) respectively.

        Returns
        -------
        output_sequence : ``torch.FloatTensor``
            The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
        final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
            The per-layer final (state, memory) states of the LSTM, with shape
            (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
            respectively. The last dimension is duplicated because it contains the state/memory
            for both the forward and backward layers.
        """
        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor,
                                               torch.Tensor]]] = [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, (forward_layer, backward_layer, state) in enumerate(zip(self.forward_layers,
                                                                                 self.backward_layers,
                                                                                 hidden_states)):
            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                                   batch_lengths,
                                                                   forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
                                                                      batch_lengths,
                                                                      backward_state)
            # Skip connections, just adding the input to the output.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence,
                                               backward_output_sequence], -1))
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor,
                                 torch.FloatTensor] = (torch.cat(final_hidden_states, 0),
                                                       torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple
