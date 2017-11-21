from typing import List, Tuple, Union, Optional
import torch

from torch.nn.utils.rnn import pack_padded_sequence
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, sort_batch_by_length

RnnState = Union[List[torch.Tensor], Tuple[torch.Tensor]]  # pylint: disable=invalid-name


class _EncoderBase(torch.nn.Module):

    def __init__(self, stateful: bool = False) -> None:
        super(_EncoderBase, self).__init__()
        self._stateful = stateful
        self._states: RnnState = None

    def sort_and_run_forward(self,
                             module: torch.nn.Module,
                             inputs: torch.Tensor,
                             mask: torch.Tensor,
                             hidden_state: Optional[RnnState] = None):

        # In some circumstances you may have sequences of zero length. ``pack_padded_sequence``
        # requires all sequence lengths to be > 0, so remove sequences of zero length before
        # calling self._module, then fill with zeros.

        # First count how many sequences are empty.
        batch_size, _ = mask.size()
        num_valid = torch.sum(mask[:, 0]).int().data[0]

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(inputs,
                                                                                           sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs[:num_valid, :, :],
                                                     sorted_sequence_lengths[:num_valid].data.tolist(),
                                                     batch_first=True)
        # Prepare the initial states.
        if not self._stateful:
            initial_states = hidden_state
        else:
            initial_states = self._get_initial_states(num_valid, batch_size)

        # Actually call the module on the sorted PackedSequence.
        packed_sequence_output, final_states = module(packed_sequence_input, initial_states)

        return packed_sequence_output, final_states, restoration_indices, num_valid

    def _get_initial_states(self,
                            num_valid: int,
                            batch_size: int) -> Union[None, Tuple[torch.Tensor], torch.Tensor]:
        # We don't know the state sizes the first time calling forward.
        if self._states is None:
            initial_states = None
        else:
            # We have some previous states.
            # It's possible this batch is larger then all previous ones.  If so, resize
            # the states.
            if batch_size > self._states[0].size(1):
                num_states_to_concat = batch_size - self._states[0].size(1)
                resized_states = []
                for state in self._states:
                    zeros = state.data.new(state.size(0), num_states_to_concat, state.size(2)).fill_(0)
                    resized_states.append(torch.cat([state, zeros], 1))
                self._states = resized_states

            if len(self._states) == 1:
                initial_states = self._states[0][:, :num_valid, :]
            else:
                initial_states = tuple([state[:, :num_valid, :]
                                        for state in self._states])
        return initial_states

    def _update_states(self,
                       final_states: Union[List[torch.Tensor], Tuple[torch.Tensor]],
                       num_valid: int,
                       restoration_indices: torch.Tensor) -> None:

        if self._states is None:
            # First time through we allocate an array to hold the states.
            states = []
            for k, state in enumerate(final_states):
                states.append(torch.autograd.Variable(state.data.new(state.size(0),
                                                                     final_states[0].size(1),
                                                                     state.size(-1)).fill_(0)))
            self._states = states

        # We may need to pad the final states for sequences of length 0.
        dim, batch_size, _ = final_states[0].size()
        if num_valid < batch_size:
            states_with_invalid_rows = []
            for state in final_states:
                zeros = state.data.new(dim, batch_size - num_valid, state.size(-1)).fill_(0)
                states_with_invalid_rows.append(torch.cat([state, zeros], 1))
        else:
            states_with_invalid_rows = list(final_states)

        for k, state in enumerate(states_with_invalid_rows):
            self._states[k].data[:, :batch_size, :] = state.index_select(
                    1, restoration_indices
            ).data
