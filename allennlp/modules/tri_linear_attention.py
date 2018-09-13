import math
import torch
from overrides import overrides
from torch.nn import Parameter


class TriLinearAttention(torch.nn.Module):
    """
    TriLinear attention as used by BiDaF, this is less flexible more memory efficient then
    the `linear` implementation since we do not create a massive
    (batch, context_len, question_len, dim) matrix
    """

    def __init__(self, input_dim: int) -> None:
        super(TriLinearAttention, self).__init__()
        self.input_dim = input_dim
        self._x_weights = Parameter(torch.Tensor(input_dim, 1))
        self._y_weights = Parameter(torch.Tensor(input_dim, 1))
        self._dot_weights = Parameter(torch.Tensor(1, 1, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.input_dim*3 + 1))
        self._y_weights.data.uniform_(-std, std)
        self._x_weights.data.uniform_(-std, std)
        self._dot_weights.data.uniform_(-std, std)


    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        batch_dim = matrix_1.size(0)
        t1 = matrix_1.size(1)
        t2 = matrix_2.size(1)

        # (batch * time1, dim) * (dim, 1) -> (batch*tim1, 1)
        x_factors = torch.matmul(matrix_1.resize(batch_dim*t1, self.input_dim), self._x_weights)
        x_factors = x_factors.contiguous().view(batch_dim, t1, 1)  # ->  (batch, tim1, 1)

        # (batch * time2, dim) * (dim, 1) -> (batch*tim2, 1)
        y_factors = torch.matmul(matrix_2.resize(batch_dim*t2, self.input_dim), self._y_weights)
        y_factors = y_factors.contiguous().view(batch_dim, 1, t2)  # ->  (batch, 1, tim2)

        weighted_x = matrix_1 * self._dot_weights  # still (batch, time1, dim)

        matrix_2_t = torch.transpose(matrix_2, 1, 2)  # -> (batch, dim, time2)

        # Batch multiplication,
        # (batch, time1, dim), (batch, dim, time2) -> (batch, time1, time2)
        dot_factors = torch.matmul(weighted_x, matrix_2_t)

        # broadcasting will correctly repeat the x/y factors as needed
        return dot_factors + x_factors + y_factors

    @classmethod
    def from_params(cls):
        raise NotImplementedError()