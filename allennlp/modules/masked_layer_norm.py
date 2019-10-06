import torch


class MaskedLayerNorm(torch.nn.Module):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, size: int, gamma0: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, size) * gamma0)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, size))
        self.size = size
        self.eps = eps

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        broadcast_mask = mask.unsqueeze(-1).float()
        num_elements = broadcast_mask.sum() * self.size
        mean = (tensor * broadcast_mask).sum() / num_elements
        masked_centered = (tensor - mean) * broadcast_mask
        std = torch.sqrt((masked_centered * masked_centered).sum() / num_elements + self.eps)
        return self.gamma * (tensor - mean) / (std + self.eps) + self.beta
