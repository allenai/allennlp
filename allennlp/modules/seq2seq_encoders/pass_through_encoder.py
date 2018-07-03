from overrides import overrides
import torch

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

@Seq2SeqEncoder.register("pass_through")
class PassThroughEncoder(Seq2SeqEncoder):
    """
    This class allows you to specify skipping a ``Seq2SeqEncoder`` just
    by changing a configuration file. This is useful for ablations and
    measuring the impact of different elements of your model.
    """
    def __init__(self, input_dim: int) -> None:
        super(PassThroughEncoder, self).__init__()
        self._input_dim = input_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        # pylint: disable=unused-argument

        return inputs
