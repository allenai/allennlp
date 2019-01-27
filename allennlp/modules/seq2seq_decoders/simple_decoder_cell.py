import torch
from typing import Tuple

from torch.nn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.modules import Attention, SimilarityFunction
from allennlp.modules.attention import LegacyAttention
from allennlp.modules.seq2seq_decoders.decoder_cell import DecoderCell
from allennlp.nn import util


@DecoderCell.register("simple_decoder")
class SimpleDecoderCell(DecoderCell):
    """
    This decoder cell implements simple decoding network with LSTMCell and Attention
        as it was implemented in ``SimpleSeq2Seq``

    Parameters
    ----------
    decoding_dim : ``int``, required
        Defines dimensionality of output vectors. Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function: ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    """

    def __init__(self,
                 decoding_dim: int,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 ):
        super(SimpleDecoderCell, self).__init__(decoding_dim=decoding_dim)

        # In this particular type of decoder output of previous step passes directly to the input of current step
        # We also assume that sequence encoder output dimensionality is equal to the encoder output dimensionality
        self._sequence_encoding_dim = self._decoding_dim
        self._decoder_input_dim = self._decoding_dim
        self._decoder_output_dim = self._decoding_dim

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_input_dim + self._sequence_encoding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.Tensor = None,
                                encoder_outputs: torch.Tensor = None,
                                encoder_outputs_mask: torch.Tensor = None) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    def forward(self,
                previous_steps_predictions: torch.Tensor,
                encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor,
                decoder_hidden: torch.Tensor,
                decoder_context: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        # shape: (group_size, output_dim)
        last_predictions_embedding = previous_steps_predictions[:, -1]

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, last_predictions_embedding), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = last_predictions_embedding

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        return decoder_hidden, decoder_context

