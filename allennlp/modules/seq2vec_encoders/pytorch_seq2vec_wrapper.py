from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


class PytorchSeq2VecWrapper(Seq2VecEncoder):
    """
    PyTorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
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

    In order to be wrapped with this wrapper, a class must have the following members:

        - ``self.input_size: int``
        - ``self.hidden_size: int``
        - ``def forward(inputs: PackedSequence, hidden_state: torch.tensor) ->
          Tuple[PackedSequence, torch.Tensor]``.
        - ``self.bidirectional: bool`` (optional)

    This is what PyTorch's RNN's look like - just make sure your class looks like those, and it
    should work.

    Note that we *require* you to pass sequence lengths when you call this module, to avoid subtle
    bugs around masking.  If you already have a ``PackedSequence`` you can pass ``None`` as the
    second parameter.
    """

    def __init__(self,
                 module: torch.nn.modules.RNNBase,
                 return_all_layers: bool = False,
                 return_all_hidden_states: bool = False) -> None:
        # Seq2VecEncoders cannot be stateful.
        super(PytorchSeq2VecWrapper, self).__init__(stateful=False)
        self._module = module
        self._return_all_layers = return_all_layers
        self._return_all_hidden_states = return_all_hidden_states
        if not getattr(self._module, 'batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")

    @property
    def _bidirectional(self):
        return getattr(self._module, 'bidirectional', False)

    def _num_directions(self):
        return 2 if self._bidirectional else 1

    @property
    def _num_layers(self):
        return getattr(self._module, 'num_layers', 1)

    @overrides
    def get_input_dim(self) -> int:
        return self._module.input_size

    @overrides
    def get_output_dim(self) -> int:
        output_dim = self._module.hidden_size * self._num_directions()

        if self._return_all_layers:
            output_dim *= self._num_layers

        return output_dim

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            if self._return_all_layers:
                state = self._module(inputs, hidden_state)[1]  # FIXME: needs reshape?
            else:
                # If a mask isn't passed, there is no padding in the batch of instances, so we can just
                # return the last sequence output as the state.  This doesn't work in the case of
                # variable length sequences, as the last state for each element of the batch won't be
                # at the end of the max sequence length, so we have to use the state of the RNN below.

                # FIXME: take 1 instead of 0, to get (state, memory) if desired.
                state = self._module(inputs, hidden_state)[0][:, -self._num_directions():, :]

            if self._return_all_hidden_states:
                if isinstance(state, tuple):
                    return torch.stack(state)
                else:
                    return state.unsqueeze(0)
            elif isinstance(state, tuple):
                return state[0]
            else:
                return state

        batch_size = mask.size(0)

        _, state, restoration_indices, = \
            self.sort_and_run_forward(self._module, inputs, mask, hidden_state)

        # Deal with the fact the LSTM state is a tuple of (state, memory).
        # For consistency, we always add one dimension to the state and later decide if to drop it.
        if isinstance(state, tuple) and self._return_all_hidden_states:
            state = torch.stack(state)
        else:
            if isinstance(state, tuple):
                state = state[0]
            state = state.unsqueeze(0)

        return self._restore_order_and_shape(batch_size, restoration_indices, state)

    def _restore_order_and_shape(self,
                                 batch_size: int,
                                 restoration_indices: torch.LongTensor,
                                 state: torch.Tensor) -> torch.Tensor:
        # `state_len` is 2 if it's an LSTM and `self._return_all_hidden_states` is true.
        state_len, num_layers_times_directions, num_valid, encoding_dim = state.size()
        # Add back invalid rows.
        if num_valid < batch_size:
            # batch size is the third dimension here, because PyTorch returns RNN state, which is possibly a tuple,
            # as a tensor of shape (num_layers * num_directions, batch_size, hidden_size)
            zeros = state.new_zeros(state_len,
                                    num_layers_times_directions,
                                    batch_size - num_valid,
                                    encoding_dim)
            state = torch.cat([state, zeros], 2)

        # Restore the original indices and return the final state of the
        # top layer. PyTorch's recurrent layers return state in the form
        # (num_layers * num_directions, batch_size, hidden_size) regardless
        # of the 'batch_first' flag, so we transpose, extract the relevant
        # layer state (both forward and backward if using bidirectional layers)
        # and we combine the hidden states in the last dimension, just after the batch size.

        # now of shape: (state_len, batch_size, num_layers * num_directions, hidden_size).
        unsorted_state = state.transpose(1, 2).index_select(1, restoration_indices)

        if not self._return_all_layers:
            # Extract the last hidden vector, including both forward and backward states
            # if the cell is bidirectional.
            unsorted_state = unsorted_state[:, :, -self._num_directions():, :]

        if self._return_all_hidden_states:
            return unsorted_state.contiguous().view([state_len, -1, self.get_output_dim()])
        else:
            return unsorted_state[0].contiguous().view([-1, self.get_output_dim()])
