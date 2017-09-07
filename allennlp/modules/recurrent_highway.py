from typing import Callable

import torch
from overrides import overrides


class RecurrentHighway(torch.nn.Module):
    """
    A `Recurrent Highway layer does a non-linear gated combination of a linear
    transformation of an RNN input and a linear transformation of the RNN output.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size,
        input_dim)``.
    state_dim: ``int``
        The dimensionality of the state of the RNN. We assume the state has shape
        ``(batch_size, state_dim)``.
    """
    def __init__(self,
                 input_dim: int,
                 state_dim: int) -> None:
        super(RecurrentHighway, self).__init__()
        self._input_dim = input_dim
        self._state_dim = state_dim
        self.input_projection = torch.nn.Linear(input_dim, 2 * state_dim)
        self.state_projection = torch.nn.Linear(state_dim, state_dim)

        # Bias the computation intially to carry forward the current state.
        self.state_projection.bias.data.fill_(1)

    @overrides
    def forward(self,
                current_input: torch.Tensor,
                current_state: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        projected_input = self.input_projection(current_input)
        projected_state = self.state_projection(current_state)
        # NOTE: if you modify this, think about whether you should modify the initialization
        # above, too.
        gate = torch.nn.functional.sigmoid(projected_input[:, (1 * self._input_dim):(2 * self._input_dim)] +
                                           projected_state)

        return gate * current_state + (1 - gate) * projected_input[:, :self._input_dim]