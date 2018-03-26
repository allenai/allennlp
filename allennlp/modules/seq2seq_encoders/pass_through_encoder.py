
import torch

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common.params import Params

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

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        # pylint: disable=unused-argument

        return inputs

    @classmethod
    def from_params(cls, params: Params) -> "PassThroughEncoder":
        input_dim = params.pop_int("input_dim")
        params.assert_empty(cls.__name__)
        return PassThroughEncoder(input_dim=input_dim)
