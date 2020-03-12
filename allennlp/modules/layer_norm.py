import torch

from allennlp.nn import util


class LayerNorm(torch.nn.Module):

    """
    An implementation of [Layer Normalization](
    https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5).

    Layer Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:

    output = (gamma * (tensor - mean) / (std + eps)) + beta

    # Parameters

    dimension : `int`, required.
        The dimension of the layer output to normalize.

    # Returns

    The normalized layer output.
    """  # noqa

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dimension))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))

    def forward(self, tensor: torch.Tensor):
        mean = tensor.mean(-1, keepdim=True)
        std = tensor.std(-1, unbiased=False, keepdim=True)
        return (
            self.gamma * (tensor - mean) / (std + util.tiny_value_of_dtype(std.dtype)) + self.beta
        )
