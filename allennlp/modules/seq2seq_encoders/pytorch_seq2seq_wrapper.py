
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
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

    Note that we *require* you to pass a binary mask of shape (batch_size, sequence_length)
    when you call this module, to avoid subtle bugs around masking.  If you already have a
    ``PackedSequence`` you can pass ``None`` as the second parameter.

    We support stateful RNNs where the final state from each batch is used as the initial
    state for the subsequent batch by passing ``stateful=True`` to the constructor.
    """
    def __init__(self, module: torch.nn.Module, stateful: bool = False) -> None:
        super(PytorchSeq2SeqWrapper, self).__init__(stateful)
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

        if self.stateful and mask is None:
            raise ValueError("Always pass a mask with stateful RNNs.")
        if self.stateful and hidden_state is not None:
            raise ValueError("Stateful RNNs provide their own initial hidden_state.")

        if mask is None:
            return self._module(inputs, hidden_state)[0]

        batch_size, total_sequence_length = mask.size()

        packed_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._module, inputs, mask, hidden_state)

        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

        num_valid = unpacked_sequence_tensor.size(0)
        # Some RNNs (GRUs) only return one state as a Tensor.  Others (LSTMs) return two.
        # If one state, use a single element list to handle in a consistent manner below.
        if not isinstance(final_states, (list, tuple)) and self.stateful:
            final_states = [final_states]

        # Add back invalid rows.
        if num_valid < batch_size:
            _, length, output_dim = unpacked_sequence_tensor.size()
            zeros = unpacked_sequence_tensor.data.new(batch_size - num_valid, length, output_dim).fill_(0)
            zeros = Variable(zeros)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)

            # The states also need to have invalid rows added back.
            if self.stateful:
                new_states = []
                for state in final_states:
                    num_layers, _, state_dim = state.size()
                    zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
                    zeros = Variable(zeros)
                    new_states.append(torch.cat([state, zeros], 1))
                final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2SeqEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.data.new(batch_size,
                                                      sequence_length_difference,
                                                      unpacked_sequence_tensor.size(-1)).fill_(0)
            zeros = Variable(zeros)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

        if self.stateful:
            self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        return unpacked_sequence_tensor.index_select(0, restoration_indices)
