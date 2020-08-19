from overrides import overrides
import torch

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("pass_through")
class PassThroughEncoder(Seq2SeqEncoder):
    """
    This class allows you to specify skipping a `Seq2SeqEncoder` just
    by changing a configuration file. This is useful for ablations and
    measuring the impact of different elements of your model.

    Registered as a `Seq2SeqEncoder` with name "pass_through".
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
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
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        # Parameters

        inputs : `torch.Tensor`, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, timesteps).

        # Returns

        A tensor of shape (batch_size, timesteps, output_dim),
        where output_dim = input_dim.
        """
        if mask is None:
            return inputs
        else:
            # We should mask out the output instead of the input.
            # But here, output = input, so we directly mask out the input.
            return inputs * mask.unsqueeze(dim=-1)
