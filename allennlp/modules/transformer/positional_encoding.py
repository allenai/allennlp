import math
import torch

from allennlp.common import FromParams
from allennlp.nn.util import get_range_vector, get_device_of


class SinusoidalPositionalEncoding(torch.nn.Module, FromParams):
    """
    Implements the frequency-based positional encoding described
    in [Attention is All you Need][0].

    Adds sinusoids of different frequencies to a `Tensor`. A sinusoid of a
    different frequency and phase is added to each dimension of the input `Tensor`.
    This allows the attention heads to use absolute and relative positions.

    The number of timescales is equal to hidden_dim / 2 within the range
    (min_timescale, max_timescale). For each timescale, the two sinusoidal
    signals sin(timestep / timescale) and cos(timestep / timescale) are
    generated and concatenated along the hidden_dim dimension.

    [0]: https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077

    # Parameters

    tensor : `torch.Tensor`
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : `float`, optional (default = `1.0`)
        The smallest timescale to use.
    max_timescale : `float`, optional (default = `1.0e4`)
        The largest timescale to use.

    # Returns

    `torch.Tensor`
        The input tensor augmented with the sinusoidal frequencies.
    """  # noqa

    def __init__(self, min_timescale: float = 1.0, max_timescale: float = 1.0e4):
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, input_tensor: torch.Tensor):
        # TODO: Another option is to specify the expected size in init, so that we can construct
        # the positional encoding beforehand, and simply add it to the input tensor in forward.
        _, timesteps, hidden_dim = input_tensor.size()
        num_timescales = hidden_dim // 2
        device = get_device_of(input_tensor)

        timestep_range = get_range_vector(timesteps, device).data.float()
        timescale_range = get_range_vector(num_timescales, device).data.float()

        log_timescale_increments = math.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / float(num_timescales - 1)
        inverse_timescales = self.min_timescale * torch.exp(
            timescale_range * -log_timescale_increments
        )

        # Broadcasted multiplication - shape (timesteps, num_timescales)
        scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
        # shape (timesteps, 2 * num_timescales)
        sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
        return input_tensor + sinusoids.unsqueeze(0)
