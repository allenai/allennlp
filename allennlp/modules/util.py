from copy import deepcopy
import torch


def replicate_layers(layer: torch.nn.Module, num_copies: int):
    """
    # Parameters
            layer (torch.nn.Module) - The torch layer that needs to be replicated.
            num_copies (int) - Number of copies to create.

    # Returns
            A ModuleList that contains `num_copies` of the `layer`.
    """
    return torch.nn.ModuleList([deepcopy(layer) for _ in range(num_copies)])
