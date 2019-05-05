import torch
from typing import Tuple, TypeVar, Generic, Dict, Any, Optional
from allennlp.nn import util
from allennlp.common import Registrable

class DecoderModule(torch.nn.Module, Registrable):
    # pylint: disable=abstract-method

    def __init__(self, decoding_dim: int, target_embedding_dim: int, is_sequential: bool):
        super(DecoderModule, self).__init__()
        self.target_embedding_dim = target_embedding_dim
        self.decoding_dim = decoding_dim
        self.is_sequential = is_sequential

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this ``DecoderModule``.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        return self.decoding_dim

    def init_decoder_state(self, encoder_out: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        raise NotImplementedError()

    def forward(self,
                previous_state: Dict[str, torch.Tensor],
                encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor,
                previous_steps_predictions: torch.Tensor,
                previous_steps_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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
        previous_state : ``Dict[str, torch.Tensor]``, required
            previous state of decoder

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
        Tuple of new decoder state and decoder output. Output should be used to generate out sequence elements

       """
        raise NotImplementedError()
