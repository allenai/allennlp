import torch

from allennlp.nn import Activation


class GatedSum(torch.nn.Module):
    """
    This `Module` represents a gated sum of two tensors `a` and `b`. Specifically:
    ```
    f = activation(W [a; b])
    out = f * a + (1 - f) * b
    ```

    # Parameters

    input_dim : `int`, required
        The dimensionality of the input. We assume the input have shape `(..., input_dim)`.
    activation : `Activation`, optional (default = `torch.nn.Sigmoid()`)
        The activation function to use.
    """

    def __init__(self, input_dim: int, activation: Activation = torch.nn.Sigmoid()) -> None:
        super().__init__()
        self.input_dim = input_dim
        self._gate = torch.nn.Linear(input_dim * 2, 1)
        self._activation = activation

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.input_dim

    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        if input_a.size() != input_b.size():
            raise ValueError("The input must have the same size.")
        if input_a.size(-1) != self.input_dim:
            raise ValueError("Input size must match `input_dim`.")
        gate_value = self._activation(self._gate(torch.cat([input_a, input_b], -1)))
        return gate_value * input_a + (1 - gate_value) * input_b
