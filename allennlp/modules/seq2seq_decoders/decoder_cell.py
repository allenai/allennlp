import torch
from typing import Tuple

from allennlp.common import Registrable


class DecoderCell(torch.nn.Module, Registrable):
    # pylint: disable=abstract-method
    """
    A ``DecoderCell`` is a ``Module`` that takes as input a result of ``Seq2SeqEncoder`` and returns a
    new sequence of vectors.

    Decoder should implement function to perform a single decoding step.

    Parameters
    ----------
    decoding_dim : ``int``, required
        Defines dimensionality of output vectors.

    target_embedding_dim : ``int``, required
        Defines dimensionality of target embeddings. Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    """

    def __init__(self, decoding_dim, target_embedding_dim):
        super(DecoderCell, self).__init__()
        self.target_embedding_dim = target_embedding_dim
        self._decoding_dim = decoding_dim

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this ``DecoderCell``.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        return self._decoding_dim

    def forward(self,
                previous_steps_predictions: torch.Tensor,
                encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor,
                decoder_hidden: torch.Tensor,
                decoder_context: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs single decoding step, returns decoder hidden state and decoder output.

        Parameters
        ----------
        previous_steps_predictions : ``torch.Tensor``, required
            Embeddings of predictions on previous step.
            Shape: (group_size, steps_count, decoder_output_dim)
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
        raise NotImplementedError()

