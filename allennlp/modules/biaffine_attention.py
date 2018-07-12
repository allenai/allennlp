import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BiaffineAttention(nn.Module):
    """
    A Biaffine attention layer.

    This layer computes two projections of its inputs in addition
    to a Biaffine projection and returns their sum.

    """
    def __init__(self, input1_dim: int, input2_dim: int, output_dim: int) -> None:
        """
        Parameters
        ----------
        input1_dim: ``int``, required.
            The dimension of the first input.
        input2_dim: ``int``, required.
            The dimension of the second input.
        output_dim: ``int``, required.
            The dimension of the output tensor.
        """
        super(BiaffineAttention, self).__init__()

        self._input1_projection = Parameter(torch.Tensor(output_dim, input1_dim))
        self._input2_projection = Parameter(torch.Tensor(output_dim, input2_dim))
        self._bias = Parameter(torch.Tensor(output_dim, 1, 1))
        self._biaffine_projection = Parameter(torch.Tensor(output_dim,
                                                           input1_dim,
                                                           input2_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._input1_projection)
        nn.init.xavier_uniform_(self._input2_projection)
        nn.init.xavier_uniform_(self._biaffine_projection)
        nn.init.constant_(self._bias, 0.)

    def forward(self, # pylint: disable=arguments-differ
                input1: torch.Tensor,
                input2: torch.Tensor,
                input1_mask: torch.Tensor = None,
                input2_mask: torch.Tensor = None):
        """
        Parameters
        ----------
        input1 : ``torch.Tensor``
            An input tensor with shape (batch_size, timesteps, input1_dim).
        input2 : ``torch.Tensor``
             An input tensor with shape (batch_size, timesteps, input2_dim).
        input1_mask : ``torch.Tensor``
            The input1 mask with shape (batch_size, timesteps).
        input2_mask : ``torch.Tensor``
            The input2 mask with shape (batch_size, timesteps).

        Returns
        -------
        A tensor with shape (batch_size, output_dim, timesteps, timesteps).
        """
        # Shape (batch_size, num_labels, timesteps, 1)
        projected_input1 = torch.matmul(self._input1_projection, input1.transpose(1, 2)).unsqueeze(3)
        # Shape (batch_size, num_labels, 1, timesteps)
        projected_input2 = torch.matmul(self._input2_projection, input2.transpose(1, 2)).unsqueeze(2)

        # Shape (batch_size, num_label, timesteps, input1_dim)
        first_biaffine = torch.matmul(input1.unsqueeze(1), self._biaffine_projection)
        # Shape (batch, output_dim, timesteps, timesteps)
        second_biaffine = torch.matmul(first_biaffine, input2.unsqueeze(1).transpose(2, 3))
        combined = second_biaffine + projected_input1 + projected_input2 + self._bias

        if input1_mask is not None:
            combined = combined * input1_mask.unsqueeze(1).unsqueeze(3)
        if input2_mask is not None:
            combined = combined * input2_mask.unsqueeze(1).unsqueeze(2)

        return combined
