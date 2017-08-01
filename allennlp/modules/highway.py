from typing import Callable

import torch
from overrides import overrides


class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * A(x) + (1 - g) *
    f(B(x))`, where :math:`A` and :math:`B` are linear transformations, :math:`f` is an
    element-wise non-linearity, and :math:`g` is an element-wise gate, computed as
    :math:`sigmoid(C(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``torch.nn.functional.relu``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 3)
                                            for _ in range(num_layers)])
        self._activation = activation

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = projected_input[:, 0 * self._input_dim : 1 * self._input_dim]
            nonlinear_part = projected_input[:, 1 * self._input_dim : 2 * self._input_dim]
            gate = projected_input[:, 2 * self._input_dim : 3 * self._input_dim]
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.nn.functional.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
