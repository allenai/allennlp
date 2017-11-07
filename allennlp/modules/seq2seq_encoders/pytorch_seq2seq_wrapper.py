import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.util import sort_batch_by_length, get_lengths_from_binary_sequence_mask


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

    Note that we *require* you to pass sequence lengths when you call this module, to avoid subtle
    bugs around masking.  If you already have a ``PackedSequence`` you can pass ``None`` as the
    second parameter.
    """
    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        super(PytorchSeq2SeqWrapper, self).__init__()
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        try:
            is_bidirectional = self._module.bidirectional
        except AttributeError:
            is_bidirectional = False
        return self._module.hidden_size * (2 if is_bidirectional else 1)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:

        if mask is None:
            return self._module(inputs, hidden_state)[0]

        # In some circumstances you may have sequences of zero length.
        # ``pack_padded_sequence`` requires all sequence lengths to be > 0, so here we
        # adjust the ``mask`` so that every sequence has length at least 1. Then after
        # running the RNN we zero out the corresponding rows in the result.

        # First count how many sequences are empty.
        batch_size, total_sequence_length = mask.size()
        num_valid = torch.sum(mask[:, 0]).int().data[0]

        # Force every sequence to be length at least one. Need to `.clone()` the mask
        # to avoid a RuntimeError from shared storage.
        if num_valid < batch_size:
            mask = mask.clone()
            mask[:, 0] = 1

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(inputs,
                                                                                           sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.data.tolist(),
                                                     batch_first=True)

        # Actually call the module on the sorted PackedSequence.
        packed_sequence_output, _ = self._module(packed_sequence_input, hidden_state)
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

        # We sorted by length, so if there are invalid rows that need to be zeroed out
        # they will be at the end.
        if num_valid < batch_size:
            unpacked_sequence_tensor[num_valid:, :, :] = 0.

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2SeqEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.data.new(batch_size, sequence_length_difference,
                                                      unpacked_sequence_tensor.size(-1)).fill_(0)
            zeros = torch.autograd.Variable(zeros)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)


        # Restore the original indices and return the sequence.
        return unpacked_sequence_tensor.index_select(0, restoration_indices)
