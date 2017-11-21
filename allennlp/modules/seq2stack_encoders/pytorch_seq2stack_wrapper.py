import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2stack_encoders.seq2stack_encoder import Seq2StackEncoder


class PytorchSeq2StackWrapper(Seq2StackEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the first one as a single output.  This
    wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from ``get_output_dim``.

    In order to be wrapped with this wrapper, a class must have the following members:

        - ``self.input_size: int``
        - ``self.hidden_size: int``
        - ``def forward(inputs: PackedSequence, hidden_state: torch.autograd.Variable) ->
          Tuple[PackedSequence, torch.autograd.Variable]``.
        - ``self.bidirectional: bool`` (optional)

    This is what pytorch's RNN's look like - just make sure your class looks like those, and it
    should work.

    Note that we *require* you to pass sequence lengths when you call this module, to avoid subtle
    bugs around masking.  If you already have a ``PackedSequence`` you can pass ``None`` as the
    second parameter.

    We support stateful RNNs where the final state from each batch is used as the initial
    state for the subsequent batch by passing ``stateful=True`` to the constructor.

    We also support stacked RNNs that return activations for each layer by passing ``stacked=True``
    to the constructor.  In this case, the ``module`` forward method has a slightly different
    signature from ``torch.nn.modules.RNNBase``.  It returns:

        - hidden states of size ``(num_layers, batch_size, timesteps, hidden_dim)``
        - final states, a tuple of sizes ``(num_layers, batch_size, hidden_dim)``
          and ``(num_layers, batch_size, memory_dim)``

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
        if not isinstance(final_states, (list, tuple)):
            final_states = [final_states]

        # Add back invalid rows.
        if num_valid < batch_size:
            _, length, dim = per_layer_sequence_outputs[0].size()
            zeros = per_layer_sequence_outputs[0].data.new(batch_size - num_valid, length, dim).fill_(0)
            for k in range(num_layers):
                per_layer_sequence_outputs[k] = torch.cat([per_layer_sequence_outputs[k], zeros], 0)

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2SeqEncoder. However, packing and unpacking
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
