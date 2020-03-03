from typing import Tuple, Dict, Optional
import torch
from allennlp.common import Registrable


class DecoderNet(torch.nn.Module, Registrable):

    """
    This class abstracts the neural architectures for decoding the encoded states and
    embedded previous step prediction vectors into a new sequence of output vectors.

    The implementations of `DecoderNet` is used by implementations of
    `allennlp.modules.seq2seq_decoders.seq_decoder.SeqDecoder` such as
    `allennlp.modules.seq2seq_decoders.seq_decoder.auto_regressive_seq_decoder.AutoRegressiveSeqDecoder`.

    The outputs of this module would be likely used by `allennlp.modules.seq2seq_decoders.seq_decoder.SeqDecoder`
    to apply the final output feedforward layer and softmax.

    # Parameters

    decoding_dim : `int`, required
        Defines dimensionality of output vectors.
    target_embedding_dim : `int`, required
        Defines dimensionality of target embeddings. Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    decodes_parallel : `bool`, required
        Defines whether the decoder generates multiple next step predictions at in a single `forward`.
    """

    def __init__(
        self, decoding_dim: int, target_embedding_dim: int, decodes_parallel: bool
    ) -> None:
        super().__init__()
        self.target_embedding_dim = target_embedding_dim
        self.decoding_dim = decoding_dim
        self.decodes_parallel = decodes_parallel

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this `DecoderNet`.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        return self.decoding_dim

    def init_decoder_state(
        self, encoder_out: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.

        # Parameters

        batch_size : `int`
            Size of batch
        final_encoder_output : `torch.Tensor`
            Last state of the Encoder

        # Returns

        `Dict[str, torch.Tensor]`
        Initial state
        """
        raise NotImplementedError()

    def forward(
        self,
        previous_state: Dict[str, torch.Tensor],
        encoder_outputs: torch.Tensor,
        source_mask: torch.BoolTensor,
        previous_steps_predictions: torch.Tensor,
        previous_steps_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        """
        Performs a decoding step, and returns dictionary with decoder hidden state or cache and the decoder output.
        The decoder output is a 3d tensor (group_size, steps_count, decoder_output_dim)
        if `self.decodes_parallel` is True, else it is a 2d tensor with (group_size, decoder_output_dim).

        # Parameters

        previous_steps_predictions : `torch.Tensor`, required
            Embeddings of predictions on previous step.
            Shape: (group_size, steps_count, decoder_output_dim)
        encoder_outputs : `torch.Tensor`, required
            Vectors of all encoder outputs.
            Shape: (group_size, max_input_sequence_length, encoder_output_dim)
        source_mask : `torch.BoolTensor`, required
            This tensor contains mask for each input sequence.
            Shape: (group_size, max_input_sequence_length)
        previous_state : `Dict[str, torch.Tensor]`, required
            previous state of decoder

        # Returns

        Tuple[Dict[str, torch.Tensor], torch.Tensor]
        Tuple of new decoder state and decoder output. Output should be used to generate out sequence elements
       """
        raise NotImplementedError()
