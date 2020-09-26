"""
A maxout neural network.
"""
from typing import Sequence, Union

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import FromParams


class Maxout(torch.nn.Module, FromParams):
    """
    This `Module` is a maxout neural network.

    # Parameters

    input_dim : `int`, required
        The dimensionality of the input.  We assume the input has shape `(batch_size, input_dim)`.
    num_layers : `int`, required
        The number of maxout layers to apply to the input.
    output_dims : `Union[int, Sequence[int]]`, required
        The output dimension of each of the maxout layers.  If this is a single `int`, we use
        it for all maxout layers.  If it is a `Sequence[int]`, `len(output_dims)` must be
        `num_layers`.
    pool_sizes : `Union[int, Sequence[int]]`, required
        The size of max-pools.  If this is a single `int`, we use
        it for all maxout layers.  If it is a `Sequence[int]`, `len(pool_sizes)` must be
        `num_layers`.
    dropout : `Union[float, Sequence[float]]`, optional (default = `0.0`)
        If given, we will apply this amount of dropout after each layer.  Semantics of `float`
        versus `Sequence[float]` is the same as with other parameters.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        output_dims: Union[int, Sequence[int]],
        pool_sizes: Union[int, Sequence[int]],
        dropout: Union[float, Sequence[float]] = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(output_dims, list):
            output_dims = [output_dims] * num_layers  # type: ignore
        if not isinstance(pool_sizes, list):
            pool_sizes = [pool_sizes] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(output_dims) != num_layers:
            raise ConfigurationError(
                "len(output_dims) (%d) != num_layers (%d)" % (len(output_dims), num_layers)
            )
        if len(pool_sizes) != num_layers:
            raise ConfigurationError(
                "len(pool_sizes) (%d) != num_layers (%d)" % (len(pool_sizes), num_layers)
            )
        if len(dropout) != num_layers:
            raise ConfigurationError(
                "len(dropout) (%d) != num_layers (%d)" % (len(dropout), num_layers)
            )

        self._pool_sizes = pool_sizes
        input_dims = [input_dim] + output_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim, pool_size in zip(
            input_dims, output_dims, pool_sizes
        ):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim * pool_size))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dims = output_dims
        self._output_dim = output_dims[-1]
        self._input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self._input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        output = inputs
        for layer, layer_output_dim, dropout, pool_size in zip(
            self._linear_layers, self._output_dims, self._dropout, self._pool_sizes
        ):
            affine_output = layer(output)
            # Compute and apply the proper shape for the max.
            shape = list(inputs.size())
            shape[-1] = layer_output_dim
            shape.append(pool_size)

            maxed_output = torch.max(affine_output.view(*shape), dim=-1)[0]
            dropped_output = dropout(maxed_output)
            output = dropped_output
        return output
