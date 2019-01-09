import torch
from typing import Tuple

from allennlp.modules import Attention, SimilarityFunction
from allennlp.modules.seq2seq_decoders.seq2seq2_decoder import Seq2SeqDecoder


@Seq2SeqDecoder.register("simple_decoder")
class SimpleSeq2SeqDecoder(Seq2SeqDecoder):
    """
    This decoder implements same decoding mechanism as was implemented in ``SimpleSeq2Seq``
    """

    def __init__(self,
                 output_dim: int,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 ):
        super(SimpleSeq2SeqDecoder, self).__init__(output_dim=output_dim)

        self._output_dim = output_dim
        self._attention_function = attention_function
        self._attention = attention

    def forward(self,
                last_steps_predictions: torch.Tensor,
                encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor,
                decoder_hidden: torch.Tensor,
                decoder_context: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs single decoding step, returns decoder hidden state and decoder output.

        Parameters
        ----------
        last_steps_predictions : ``torch.Tensor``, required
            Embeddings of all predictions on previous steps.
            Shape: (group_size, steps_done, target_embedding_dim)
        encoder_outputs : ``torch.Tensor``, required
            Vectors of all encoder outputs.
            Shape: (group_size, max_input_sequence_length, encoder_output_dim)
        source_mask : ``torch.Tensor``, required
            This tensor contains mask for each input sequence.
            Shape: (group_size, max_input_sequence_length)
        decoder_hidden : ``torch.Tensor``, required
            Decoder hidden state, generated on previous decoding step.
            Shape: (group_size, decoder_output_dim)
        decoder_context : ``torch.Tensor``, required
            Decoder output, generated on previous decoding step.
            Shape: (group_size, decoder_output_dim)

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(decoder_hidden, decoder_context)``
        """
        pass

