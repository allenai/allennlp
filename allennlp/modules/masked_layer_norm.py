import torch

from allennlp.nn import util


class MaskedLayerNorm(torch.nn.Module):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, size: int, gamma0: float = 0.1) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, size) * gamma0)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, size))
        self.size = size

    def forward(self, tensor: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:

        broadcast_mask = mask.unsqueeze(-1)
        num_elements = broadcast_mask.sum() * self.size
        mean = (tensor * broadcast_mask).sum() / num_elements
        masked_centered = (tensor - mean) * broadcast_mask
        std = torch.sqrt(
            (masked_centered * masked_centered).sum() / num_elements
            + util.tiny_value_of_dtype(tensor.dtype)
        )
        return (
            self.gamma * (tensor - mean) / (std + util.tiny_value_of_dtype(tensor.dtype))
            + self.beta
        )
