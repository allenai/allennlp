import torch
from torch.nn.utils.rnn import pack_padded_sequence

from allennlp.common.tensor import sort_batch_by_length
from allennlp.modules import Seq2VecEncoder
from allennlp.common.checks import ConfigurationError


class PytorchSeq2VecWrapper(Seq2VecEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the second one as a single output.  This
    wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from ``get_output_dim``.

    Also, there are lots of ways you could imagine going from an RNN hidden state at every
    timestep to a single vector - you could take the last vector at all layers in the stack, do
    some kind of pooling, take the last vector of the top layer in a stack, or many other  options.
    We just take the final hidden state vector, or in the case of a bidirectional RNN cell, we
    concatenate the forward and backward final states together. TODO(mattg): allow for other ways
    of wrapping RNNs.
    """
    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        super(PytorchSeq2VecWrapper, self).__init__()
        self._module = module
        if not self._module.batch_first:
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")

    def get_output_dim(self) -> int:
        return self._module.hidden_size * (2 if self._module.bidirectional else 1)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                sequence_lengths: torch.LongTensor = None) -> torch.Tensor:

        if sequence_lengths is None:
            # If sequence_lengths aren't passed, there is no padding in the batch of
            # instances, so we can just return the last sequence output as the state.
            # This doesn't work in the case of variable length sequences, as the last
            # state for each element of the batch won't be at the end of the max sequence
            # length, so we have to use the state of the RNN below.
            return self._module(inputs)[0][:, -1, :]
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(inputs,
                                                                                           sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.tolist(),
                                                     batch_first=True)

        # Actually call the module on the sorted PackedSequence.
        _, state = self._module(packed_sequence_input)

        # Deal with the fact the LSTM state is a tuple of (state, memory).
        if isinstance(state, tuple):
            state = state[0]

        # Restore the original indices and return the final state of the
        # top layer. Pytorch's recurrent layers return state in the form
        # (num_layers * num_directions, batch_size, hidden_size) regardless
        # of the 'batch_first' flag, so we transpose, extract the relevant
        # layer state (both forward and backward if using bidirectional layers)
        # and return them as a single (batch_size, self.get_output_dim()) tensor.

        # now of shape: (batch_size, num_layers * num_directions, hidden_size).
        unsorted_state = state.transpose(0, 1)[restoration_indices]

        # Extract the last hidden vector, including both forward and backward states
        # if the cell is bidirectional. Then reshape by concatenation (in the case
        # we have bidirectional states) or just squash the 1st dimension in the non-
        # bidirectional case. Return tensor has shape (batch_size, hidden_size * num_directions).
        last_state_index = 2 if self._module.bidirectional else 1
        last_layer_state = unsorted_state[:, -last_state_index:, :]
        return last_layer_state.contiguous().view([-1, self.get_output_dim()])
