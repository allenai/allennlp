from typing import Sequence, List
import math

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers: Sequence[Sequence[int]],
        direction: str,
        do_weight_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self._convolutions = torch.nn.ModuleList()
        last_dim = input_dim
        for k, layer in enumerate(layers):
            # We run two convolutions for each block -- one for the
            # output and one for the gates -- do them at once, and
            # we'll worry about slicing them in forward
            if len(layer) == 2:
                # no dilation
                conv = torch.nn.Conv1d(
                    last_dim, layer[1] * 2, layer[0], stride=1, padding=layer[0] - 1, bias=True
                )
            elif len(layer) == 3:
                # a dilation
                assert layer[0] == 2, "only support kernel = 2 for now"
                conv = torch.nn.Conv1d(
                    last_dim,
                    layer[1] * 2,
                    layer[0],
                    stride=1,
                    padding=layer[2],
                    dilation=layer[2],
                    bias=True,
                )
            else:
                raise ValueError("each layer must have length 2 or 3")

            # from Convolutional Sequence to Sequence Learning
            if k == 0:
                conv_dropout = dropout
            else:
                # no dropout
                conv_dropout = 0.0
            std = math.sqrt((4 * (1.0 - conv_dropout)) / (layer[0] * last_dim))

            conv.weight.data.normal_(0, std=std)
            conv.bias.data.zero_()

            if do_weight_norm:
                # conv.weight.shape == (out_channels, in_channels, kernel width)
                # in fairseq, conv.weight.shape == ([width, in, out])
                #   for ConvTBC.  In ConvTBC, weight norm is applied as
                #   nn.utils.weight_norm(m, dim=2) over the output dimension.
                # so for regular 1D convs we need to apply over dimension=0
                conv = torch.nn.utils.weight_norm(conv, name="weight", dim=0)

            self._convolutions.append(conv)
            last_dim = layer[1]

        assert last_dim == input_dim

        if direction not in ("forward", "backward"):
            raise ConfigurationError(f"invalid direction: {direction}")
        self._direction = direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = (batch_size, dim, timesteps)
        # outputs: (batch_size, dim, timesteps) = f(x) + x
        out = x
        timesteps = x.size(2)
        for k, convolution in enumerate(self._convolutions):
            if k == 0 and self.dropout > 0:
                # apply dropout to the input
                out = torch.nn.functional.dropout(out, self.dropout, self.training)

            conv_out = convolution(out)

            # remove the padding indices
            # x is padded by convolution width - 1 in each direction
            dims_to_remove = conv_out.size(2) - timesteps
            if dims_to_remove > 0:
                if self._direction == "forward":
                    # remove from the end of the sequence
                    conv_out = conv_out.narrow(2, 0, timesteps)
                else:
                    # remove from the beginning of the sequence
                    conv_out = conv_out.narrow(2, dims_to_remove, timesteps)

            out = torch.nn.functional.glu(conv_out, dim=1)

        # see Convolutional Sequence to Sequence Learning
        return (out + x) * math.sqrt(0.5)


@Seq2SeqEncoder.register("gated-cnn-encoder")
class GatedCnnEncoder(Seq2SeqEncoder):
    """
    **This is work-in-progress and has not been fully tested yet. Use at your own risk!**

    A `Seq2SeqEncoder` that uses a Gated CNN.

    see

    Language Modeling with Gated Convolutional Networks,  Yann N. Dauphin et al, ICML 2017
    https://arxiv.org/abs/1612.08083

    Convolutional Sequence to Sequence Learning, Jonas Gehring et al, ICML 2017
    https://arxiv.org/abs/1705.03122

    Some possibilities:

    Each element of the list is wrapped in a residual block:
    input_dim = 512
    layers = [ [[4, 512]], [[4, 512], [4, 512]], [[4, 512], [4, 512]], [[4, 512], [4, 512]]
    dropout = 0.05

    A "bottleneck architecture"
    input_dim = 512
    layers = [ [[4, 512]], [[1, 128], [5, 128], [1, 512]], ... ]

    An architecture with dilated convolutions
    input_dim = 512
    layers = [
    [[2, 512, 1]], [[2, 512, 2]], [[2, 512, 4]], [[2, 512, 8]],   # receptive field == 16
    [[2, 512, 1]], [[2, 512, 2]], [[2, 512, 4]], [[2, 512, 8]],   # receptive field == 31
    [[2, 512, 1]], [[2, 512, 2]], [[2, 512, 4]], [[2, 512, 8]],   # receptive field == 46
    [[2, 512, 1]], [[2, 512, 2]], [[2, 512, 4]], [[2, 512, 8]],   # receptive field == 57
    ]

    Registered as a `Seq2SeqEncoder` with name "gated-cnn-encoder".

    # Parameters

    input_dim : `int`, required
        The dimension of the inputs.
    layers : `Sequence[Sequence[Sequence[int]]]`, required
        The layer dimensions for each `ResidualBlock`.
    dropout : `float`, optional (default = `0.0`)
        The dropout for each `ResidualBlock`.
    return_all_layers : `bool`, optional (default = `False`)
        Whether to return all layers or just the last layer.
    """

    def __init__(
        self,
        input_dim: int,
        layers: Sequence[Sequence[Sequence[int]]],
        dropout: float = 0.0,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()

        self._forward_residual_blocks = torch.nn.ModuleList()
        self._backward_residual_blocks = torch.nn.ModuleList()
        self._input_dim = input_dim
        self._output_dim = input_dim * 2

        for layer in layers:
            self._forward_residual_blocks.append(
                ResidualBlock(input_dim, layer, "forward", dropout=dropout)
            )
            self._backward_residual_blocks.append(
                ResidualBlock(input_dim, layer, "backward", dropout=dropout)
            )

        self._return_all_layers = return_all_layers

    def forward(self, token_embeddings: torch.Tensor, mask: torch.BoolTensor):

        # Convolutions need transposed input
        transposed_embeddings = torch.transpose(token_embeddings, 1, 2)

        # We need to broadcast the mask to feature dimension,
        # and to use masked_fill_ we need the inverse of the mask.
        mask_for_fill = ~mask.unsqueeze(1)

        if self._return_all_layers:
            # outputs will be [[all forward layers], [all backward layers]]
            layer_outputs: List[List[torch.Tensor]] = [[], []]
        else:
            # outputs will be [forward final layer, backward final layer]
            outputs: List[torch.Tensor] = []

        for k, blocks in enumerate([self._forward_residual_blocks, self._backward_residual_blocks]):
            out = transposed_embeddings
            # Due to zero padding for backward sequences, we need
            # to ensure that the input has zeros everywhere where
            # there isn't a mask.
            for block in blocks:
                out = block(out.masked_fill(mask_for_fill, 0.0))
                if self._return_all_layers:
                    layer_outputs[k].append(out)
            if not self._return_all_layers:
                outputs.append(out)

        if self._return_all_layers:
            return [
                torch.cat([fwd, bwd], dim=1).transpose(1, 2) for fwd, bwd in zip(*layer_outputs)
            ]
        else:
            # Concatenate forward and backward, then transpose back
            return torch.cat(outputs, dim=1).transpose(1, 2)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def is_bidirectional(self) -> bool:
        return True
