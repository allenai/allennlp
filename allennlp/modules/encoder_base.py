from typing import Tuple, Union, Optional
import torch

from torch.nn.utils.rnn import pack_padded_sequence
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, sort_batch_by_length

RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # pylint: disable=invalid-name


class _EncoderBase(torch.nn.Module):
    # pylint: disable=abstract-method
    """
    This abstract class serves as a base for the 3 ``Encoder`` abstractions in AllenNLP.
    - :class:`~allennlp.modules.seq2seq_encoders.Seq2SeqEncoders`
    - :class:`~allennlp.modules.seq2vec_encoders.Seq2VecEncoders`
    - :class:`~allennlp.modules.seq2stack_encoders.Seq2StackEncoders`

    These classes can be inherited by any :class:`~torch.nn.Module` which implements
    one of these APIs.

    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """

    def __init__(self, stateful: bool = False) -> None:
        super(_EncoderBase, self).__init__()
        self._stateful = stateful
        self._states: RnnState = None

    def sort_and_run_forward(self,
                             module: torch.nn.Module,
                             inputs: torch.Tensor,
                             mask: torch.Tensor,
                             hidden_state: Optional[RnnState] = None):
        """
        This function exists because Pytorch RNNs require that their inputs are sorted
        before being passed as input. As all of our Seq2xxxEncoders use this functionality,
        it is provided in a base class. This method can be called on any module which
        takes as input a ``PackedSequence`` and some hidden_state, which can either be a
        tuple of tensors or a tensor.

        As all of our Seq2xxxEncoders have different return types, we return `sorted`
        outputs from the module which is called directly. Additionally, we return the
        indices into the batch dimension required to restore the tensor to it's correct,
        unsorted order and the number of valid batch elements (i.e the number of elements
        in the batch which are not completely masked). This un-sorting and re-padding
        of the module outputs is left to the subclasses because their outputs have different
        types and handling them smoothly here is difficult.

        Parameters
        ----------
        module : ``torch.nn.Module``, required.
            The module to run on the inputs.
        inputs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, embedding_size) representing
            the inputs to the Encoder.
        mask : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length), representing masked and
            non-masked elements of the sequence for each element in the batch.
        hidden_state : ``Optional[RnnState]``, (default = None).

        Returns
        -------
        module_output : ``Union[torch.Tensor, PackedSequence]``.
            A Tensor or PackedSequence representing the output of the Pytorch Module.
        final_states : ``Optional[RnnState]``
            A Tensor representing the hidden state of the Pytorch Module. This can either
            be a single tensor of shape (num_layers, batch_size, encoding_size), for instance in
            the case of a GRU, or a tuple of tensors, such as those required for an LSTM.
        restoration_indices : ``torch.LongTensor``
            A tensor of shape (batch_size,), describing the re-indexing required to transform
            the outputs back to their original batch order.
        num_valid : ``int``
            The number of valid batches used in the call to the module. This is returned so
            that the outputs of the module can be padded back to their original batch shape,
            if we altered it due to fully padded batch elements.
        """
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
        module_output, final_states = module(packed_sequence_input, initial_states)

        return module_output, final_states, restoration_indices, num_valid

    def _get_initial_states(self,
                            num_valid: int,
                            batch_size: int) -> Optional[RnnState]:
        """

        Parameters
        ----------
        num_valid : ``int``, required.
            As we can encounter batches which have completely padded rows, we only need
            to fetch the state for the elements in the batch which are used. This represents

        Returns
        -------
        """
        # We don't know the state sizes the first time calling forward.
        if self._states is None:
            initial_states = None
        else:
            # We have some previous states.
            # It's possible this batch is larger then all previous ones.
            # If so, resize the states.
            if batch_size > self._states[0].size(1):
                num_states_to_concat = batch_size - self._states[0].size(1)
                resized_states = []
                for state in self._states:
                    zeros = state.data.new(state.size(0),
                                           num_states_to_concat,
                                           state.size(2)).fill_(0)
                    resized_states.append(torch.cat([state, zeros], 1))
                self._states = resized_states

            if len(self._states) == 1:
                initial_states = self._states[0][:, :num_valid, :]
            else:
                initial_states = tuple([state[:, :num_valid, :] for state in self._states])
        return initial_states

    def _update_states(self,
                       final_states: RnnState,
                       num_valid: int,
                       restoration_indices: torch.Tensor) -> None:
        # TODO(Mark): This is wrong, it's returning unsorted states which don't then get sorted.
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

        # TODO(Mark): This is wrong, it's zeroing out previous states.
        for k, state in enumerate(states_with_invalid_rows):
            self._states[k].data[:, :batch_size, :] = state.index_select(1, restoration_indices).data
