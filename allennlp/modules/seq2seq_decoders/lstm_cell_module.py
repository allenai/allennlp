import torch
from typing import Tuple, Generic, Dict, Any, Optional

from torch.nn import LSTMCell

from allennlp.modules import Attention
from allennlp.modules.seq2seq_decoders.decoder_module import DecoderModule
from allennlp.nn import util


@DecoderModule.register("lstm_cell")
class LstmCellModule(DecoderModule):
    """
    This decoder cell implements simple decoding network with LSTMCell and Attention
        as it was implemented in ``ComposedSeq2Seq``

    Parameters
    ----------
    decoding_dim : ``int``, required
        Defines dimensionality of output vectors.
    target_embedding_dim : ``int``, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    """

    def __init__(self,
                 decoding_dim: int,
                 target_embedding_dim: int,
                 attention: Attention = None,
                 bidirectional_input: bool = False,
                 ):
        super(LstmCellModule, self).__init__(
            decoding_dim=decoding_dim,
            target_embedding_dim=target_embedding_dim,
            is_sequential=True
        )

        # In this particular type of decoder output of previous step passes directly to the input of current step
        # We also assume that sequence decoder output dimensionality is equal to the encoder output dimensionality
        self._sequence_encoding_dim = self._decoding_dim
        self._decoder_input_dim = self.target_embedding_dim
        self._decoder_output_dim = self._decoding_dim

        # Attention mechanism applied to the encoder output for each step.
        self._attention: Attention = attention

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_input_dim + self._sequence_encoding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._bidirectional_input = bidirectional_input

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

    def init_decoder_state(self, encoder_out: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        batch_size, _ = encoder_out["source_mask"].size()

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            encoder_out["encoder_outputs"],
            encoder_out["source_mask"],
            bidirectional=self._bidirectional_input)

        return {
            "decoder_hidden": final_encoder_output,  # shape: (batch_size, decoder_output_dim)
            "decoder_context": final_encoder_output.new_zeros(
                batch_size,
                self._decoder_output_dim
            )  # shape: (batch_size, decoder_output_dim)
        }

    def forward(self,
                previous_state: Dict[str, torch.Tensor],
                encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor,
                previous_steps_predictions: torch.Tensor,
                previous_steps_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        decoder_hidden = previous_state['decoder_hidden']
        decoder_context = previous_state['decoder_context']

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

        return {"decoder_hidden": decoder_hidden, "decoder_context": decoder_context}, decoder_hidden
