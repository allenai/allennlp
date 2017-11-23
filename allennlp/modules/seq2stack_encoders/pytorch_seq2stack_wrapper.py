import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2stack_encoders.seq2stack_encoder import Seq2StackEncoder


class PytorchSeq2StackWrapper(Seq2StackEncoder):
    """
    This class wraps Pytorch-like RNN modules which take as input a ``PackedSequence`` and some
    hidden state and return all outputs of all layers of the stacked encoder as a tensor of shape
    (num_layers, batch_size, sequence_length, encoding_size). Although no Pytorch RNNs actually
    look like this, it's easier if we fit the API for the inputs to these modules, so all of the
    Encoders in AllenNLP have a unified interface for at least their inputs.

    In order to be wrapped with this wrapper, a class must have the following members:

        - ``self.input_size: int``
        - ``self.hidden_size: int``
        - ``def forward(inputs: PackedSequence, hidden_state: torch.autograd.Variable) ->
          Tuple[torch.FloatTensor, torch.autograd.Variable]``.
        - ``self.bidirectional: bool`` (optional)


    This class also supports stateful RNNs where the final state from each batch is used as the
    initial state for the subsequent batch by passing ``stateful=True`` to the constructor.
    In this case, the ``module.forward`` method has a slightly different signature from
    ``torch.nn.modules.RNNBase``.  It returns:

        - a  of size ``(num_layers, batch_size, timesteps, hidden_dim)``
        - final states, a tuple of sizes ``(num_layers, batch_size, hidden_dim)``
          and ``(num_layers, batch_size, memory_dim)``

    This class then handles sorting, statefulness and padding before returning the stacked
    hidden_states.
    """
    def __init__(self,
                 module: torch.nn.Module,
                 stateful: bool = False) -> None:
        super(PytorchSeq2StackWrapper, self).__init__(stateful)
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

        try:
            is_bidirectional = self._module.bidirectional
        except AttributeError:
            is_bidirectional = False
        if is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._num_directions

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:

        if self._stateful and mask is None:
            raise ValueError("Always pass a mask with stateful RNNs.")
        if self._stateful and hidden_state is not None:
            raise ValueError("Stateful RNNs provide their own initial hidden_state.")

        if mask is None:
            return self._module(inputs, hidden_state)[0]

        batch_size, total_sequence_length = mask.size()

        stacked_sequence_output, final_states, restoration_indices, num_valid = \
            self.sort_and_run_forward(self._module, inputs, mask, hidden_state)

        # stacked_sequence_output is shape (num_layers, batch_size, timesteps, encoder_dim)
        num_layers = stacked_sequence_output.size(0)
        per_layer_sequence_outputs = [layer.squeeze(0) for layer in
                                      stacked_sequence_output.chunk(num_layers, 0)]

        # Some RNNs (GRUs) only return one state as a Tensor.  Others (LSTMs) return two.
        # If one state, use a single element list to handle in a consistent manner below.
        if not isinstance(final_states, (list, tuple)) and self._stateful:
            final_states = [final_states]

        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        if num_valid < batch_size:
            _, length, dim = per_layer_sequence_outputs[0].size()
            zeros = per_layer_sequence_outputs[0].data.new(batch_size - num_valid,
                                                           length, dim).fill_(0)
            for k in range(num_layers):
                per_layer_sequence_outputs[k] = torch.cat([per_layer_sequence_outputs[k],
                                                           zeros], 0)

            # The states also need to have invalid rows added back.
            if self._stateful:
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

        if self._stateful:
            self._update_states(final_states, num_valid, restoration_indices)

        # Restore the original indices and return the sequence.
        return torch.cat([tensor.index_select(0, restoration_indices).unsqueeze(0)
                          for tensor in per_layer_sequence_outputs], dim=0)
