from overrides import overrides
import torch

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules import FeedForward


@Seq2SeqEncoder.register("feedforward")
class FeedForwardEncoder(Seq2SeqEncoder):
    """
    This class applies the `FeedForward` to each item in sequences.
    """
    def __init__(self, feedforward: FeedForward) -> None:
        super().__init__()
        self._feedforward = feedforward

    @overrides
    def get_input_dim(self) -> int:
        return self._feedforward.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._feedforward.get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        return self._feedforward(inputs)
