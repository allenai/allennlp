from typing import List

import torch

from allennlp.nn import util


class RnnStatelet:
    """
    This class keeps track of all of decoder-RNN-related variables that you need during decoding.
    This includes things like the current decoder hidden state, the memory cell (for LSTM
    decoders), the encoder output that you need for computing attentions, and so on.

    This is intended to be used `inside` a ``State``, which likely has other things it has to keep
    track of for doing constrained decoding.

    Parameters
    ----------
    hidden_state : ``torch.Tensor``
        This holds the LSTM hidden state, with shape ``(decoder_output_dim,)`` if the decoder
        has 1 layer and ``(num_layers, decoder_output_dim)`` otherwise.
    memory_cell : ``torch.Tensor``
        This holds the LSTM memory cell, with shape ``(decoder_output_dim,)`` if the decoder has
        1 layer and ``(num_layers, decoder_output_dim)`` otherwise.
    previous_action_embedding : ``torch.Tensor``
        This holds the embedding for the action we took at the last timestep (which gets input to
        the decoder).  Has shape ``(action_embedding_dim,)``.
    attended_input : ``torch.Tensor``
        This holds the attention-weighted sum over the input representations that we computed in
        the previous timestep.  We keep this as part of the state because we use the previous
        attention as part of our decoder cell update.  Has shape ``(encoder_output_dim,)``.
    encoder_outputs : ``List[torch.Tensor]``
        A list of variables, each of shape ``(input_sequence_length, encoder_output_dim)``,
        containing the encoder outputs at each timestep.  The list is over batch elements, and we
        do the input this way so we can easily do a ``torch.cat`` on a list of indices into this
        batched list.

        Note that all of the above parameters are single tensors, while the encoder outputs and
        mask are lists of length ``batch_size``.  We always pass around the encoder outputs and
        mask unmodified, regardless of what's in the grouping for this state.  We'll use the
        ``batch_indices`` for the group to pull pieces out of these lists when we're ready to
        actually do some computation.
    encoder_output_mask : ``List[torch.Tensor]``
        A list of variables, each of shape ``(input_sequence_length,)``, containing a mask over
        question tokens for each batch instance.  This is a list over batch elements, for the same
        reasons as above.
    """
    def __init__(self,
                 hidden_state: torch.Tensor,
                 memory_cell: torch.Tensor,
                 previous_action_embedding: torch.Tensor,
                 attended_input: torch.Tensor,
                 encoder_outputs: List[torch.Tensor],
                 encoder_output_mask: List[torch.Tensor]) -> None:
        self.hidden_state = hidden_state
        self.memory_cell = memory_cell
        self.previous_action_embedding = previous_action_embedding
        self.attended_input = attended_input
        self.encoder_outputs = encoder_outputs
        self.encoder_output_mask = encoder_output_mask

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return all([
                    util.tensors_equal(self.hidden_state, other.hidden_state, tolerance=1e-5),
                    util.tensors_equal(self.memory_cell, other.memory_cell, tolerance=1e-5),
                    util.tensors_equal(self.previous_action_embedding, other.previous_action_embedding,
                                       tolerance=1e-5),
                    util.tensors_equal(self.attended_input, other.attended_input, tolerance=1e-5),
                    util.tensors_equal(self.encoder_outputs, other.encoder_outputs, tolerance=1e-5),
                    util.tensors_equal(self.encoder_output_mask, other.encoder_output_mask, tolerance=1e-5),
                    ])
        return NotImplemented
